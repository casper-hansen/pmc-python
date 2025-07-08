# uv venv
# source .venv/bin/activate
# uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --torch-backend=cu128
# VLLM_USE_V1=1 vllm serve Qwen/Qwen3-Embedding-4B --task embed --disable-log-requests -q fp8 --max-num-batched-tokens 65536 --data-parallel-size $(nvidia-smi -L | wc -l)
import asyncio
import hashlib
import json
import os
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

MAX_CONCURRENT_REQUESTS = 200
LOADED_BATCH_SIZE = 1000
MAX_LENGTH = 32768
CACHE_PATH = "data/embedding_cache.jsonl"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
CACHE_WRITE_LOCK = asyncio.Lock()

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test-key",
)

cache: dict[str, list[float]] = {}

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as fh:
        for line in fh:
            try:
                record = json.loads(line)
                cache[record["hash"]] = record["embedding"]
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., partial writes)
                continue
    print(f"Loaded {len(cache)} cached embeddings from {CACHE_PATH}.")


async def embed_text(text: str, sem: asyncio.Semaphore, lock: asyncio.Lock):
    """Return embedding for *text*, using cache when available.

    If the embedding is not cached, it is requested from the model, then stored
    immediately so progress persists across crashes/restarts.
    """
    key = hashlib.md5(text.encode("utf-8")).hexdigest()

    # Reuse from cache if we already have it.
    if key in cache:
        return cache[key]

    # Otherwise, fetch from model (respect concurrency limit).
    async with sem:
        response = await client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        embedding = response.data[0].embedding

    # Persist to cache (file + in-memory) so we don't lose progress.
    async with lock:
        cache[key] = embedding
        # Append in JSONL format so we avoid rewriting the whole file.
        with open(CACHE_PATH, "a") as fh:
            fh.write(json.dumps({"hash": key, "embedding": embedding}) + "\n")

    return embedding


async def main():
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown",
        split="train",
        num_proc=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    ds = ds.filter(
        lambda batch: [
            n_tokens <= MAX_LENGTH
            for n_tokens in tokenizer(
                batch["text"],
                add_special_tokens=False,
                return_length=True,
            )["length"]
        ],
        batched=True,
        desc="Filtering long samples",
        num_proc=64,
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    num_rows = len(ds)
    num_batches = (num_rows + LOADED_BATCH_SIZE - 1) // LOADED_BATCH_SIZE

    embeddings: list[list[float]] = []

    # Outer progress bar for batches.
    for batch_idx in tqdm(range(num_batches), desc=f"Batches (samples={num_rows})", unit="batch"):
        start = batch_idx * LOADED_BATCH_SIZE
        end = min(start + LOADED_BATCH_SIZE, num_rows)
        batch = ds.select(range(start, end))

        # Spawn embedding tasks for this batch only.
        tasks = [
            asyncio.create_task(
                embed_text(text, semaphore, CACHE_WRITE_LOCK)
            )
            for text in batch["text"]
        ]

        # Inner progress bar for the current batch.
        batch_embeddings = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Embeddings (batch {batch_idx + 1}/{num_batches})",
            leave=False,
        )

        embeddings.extend(batch_embeddings)

    # Attach embeddings to dataset once everything is complete.
    ds = ds.add_column("embed", embeddings)

    print(ds)
    print("Embeddings done!")

    # Push the processed dataset to the Hugging Face Hub as the "train" split
    dataset_dict = DatasetDict({"train": ds})
    # Replace the repo_id below with your own namespace or desired repo
    dataset_dict.push_to_hub("casperhansen/pmc-oa-markdown-embeddings")


if __name__ == "__main__":
    asyncio.run(main())
