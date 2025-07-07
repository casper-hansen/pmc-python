# pip install vllm datasets
# vllm serve Qwen/Qwen3-Embedding-8B --task embed --disable-log-requests
import asyncio
import hashlib
import json
import os
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENT_REQUESTS = 50
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
    # Load dataset (consider streaming for very large datasets)
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown",
        split="train",
        num_proc=8,
    )
    ds = ds.take(1000)
    ds = ds.filter(
        lambda batch: [len(text) // 4 <= MAX_LENGTH for text in batch["text"]],
        batched=True,
        desc="Removing samples that are too long",
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    global cache_write_lock
    if cache_write_lock is None:
        cache_write_lock = asyncio.Lock()

    # Generate embeddings (cached entries return almost instantly)
    tasks = [
        asyncio.create_task(embed_text(item["text"], semaphore, cache_write_lock))
        for item in ds
    ]
    embeddings = await tqdm_asyncio.gather(*tasks)
    ds = ds.add_column("embed", embeddings)

    print(ds)
    print("Embeddings done!")


if __name__ == "__main__":
    asyncio.run(main())
