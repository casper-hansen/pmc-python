# pip install vllm datasets
# vllm serve Qwen/Qwen3-Embedding-8B --task embed --disable-log-requests
import asyncio
import hashlib
import json
import os
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENT_REQUESTS = 50
LOADED_BATCH_SIZE = 1000
MAX_LENGTH = 32768
CACHE_PATH = "data/embedding_cache.jsonl"
cache_write_lock: asyncio.Lock | None = None

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
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
            model="Qwen/Qwen3-Embedding-8B",
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
    ds = ds.filter(
        lambda batch: [len(text) // 4 <= MAX_LENGTH for text in batch["text"]],
        batched=True,
        desc="Filtering long samples",
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    global cache_write_lock
    if cache_write_lock is None:
        cache_write_lock = asyncio.Lock()

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
                embed_text(text, semaphore, cache_write_lock)
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


if __name__ == "__main__":
    asyncio.run(main())
