# uv venv
# source .venv/bin/activate
import asyncio
import hashlib
import json
import os
from tqdm import tqdm
from typing import List, Dict
from openai import AsyncOpenAI, BadRequestError
from datasets import load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

MAX_CONCURRENT_REQUESTS = 1
LOADED_BATCH_SIZE = 1000
MAX_COMPLETION_LENGTH = 4096
MAX_SAMPLE_LENGTH = 131072 - MAX_COMPLETION_LENGTH
CACHE_PATH = "data/qa_cache.jsonl"
MODEL = "qwen/qwen3-235b-a22b-07-25:free"
TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CACHE_WRITE_LOCK = asyncio.Lock()

SYSTEM_PROMPT = """\
You are a helpful assistant that summarizes the content of multiple articles in one sentence.
"""

client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "test-key"),
)

cache: dict[str, list[float]] = {}

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as fh:
        for line in fh:
            try:
                record = json.loads(line)
                cache[record["hash"]] = record["response"]
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., partial writes)
                continue
    print(f"Loaded {len(cache)} cached completions from {CACHE_PATH}.")


async def create_completion(messages: List[Dict], sem: asyncio.Semaphore, lock: asyncio.Lock):
    """Return completion for *text*, using cache when available.

    If the completion is not cached, it is requested from the model, then stored
    immediately so progress persists across crashes/restarts.
    """
    key = hashlib.md5(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()

    # Reuse from cache if we already have it.
    if key in cache:
        return cache[key]

    # Otherwise, fetch from model (respect concurrency limit).
    async with sem:
        try:
            response = await client.chat.completions.create(
                messages=messages,
                model=MODEL,
                max_completion_tokens=4096,
            )
            response = response.choices[0].message.content
        except BadRequestError as ex:
            print(ex)
            response = ""

    # Persist to cache (file + in-memory) so we don't lose progress.
    async with lock:
        cache[key] = response
        # Append in JSONL format so we avoid rewriting the whole file.
        with open(CACHE_PATH, "a") as fh:
            fh.write(json.dumps({"hash": key, "response": response}, ensure_ascii=False) + "\n")

    return response


async def main():
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown-clustering",
        split="train",
        num_proc=8,
    ).take(10)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    def _join(texts):
        return "\n\n".join(f"Article {i+1}\n{text}" for i, text in enumerate(texts))
    ds = ds.filter(
        lambda batch: [
            n_tokens + 20 <= MAX_SAMPLE_LENGTH
            for n_tokens in tokenizer(
                [SYSTEM_PROMPT + _join(texts) for texts in batch["texts"]],
                add_special_tokens=False,
                return_length=True,
            )["length"]
        ],
        batched=True,
        desc="Filtering long samples",
        num_proc=8,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    num_rows = len(ds)
    num_batches = (num_rows + LOADED_BATCH_SIZE - 1) // LOADED_BATCH_SIZE

    responses: list[list[float]] = []

    # Outer progress bar for batches.
    for batch_idx in tqdm(range(num_batches), desc=f"Batches (samples={num_rows})", unit="batch"):
        start = batch_idx * LOADED_BATCH_SIZE
        end = min(start + LOADED_BATCH_SIZE, num_rows)
        batch = ds.select(range(start, end))

        batch_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _join(texts)}
            ]
            for texts in batch["texts"]
        ]

        # Spawn completion tasks for this batch only.
        tasks = [
            asyncio.create_task(
                create_completion(messages, semaphore, CACHE_WRITE_LOCK)
            )
            for messages in batch_messages
        ]

        # Inner progress bar for the current batch.
        batch_responses = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Completions (batch {batch_idx + 1}/{num_batches})",
            leave=False,
        )
        responses.extend(batch_responses)
    
    ds = ds.add_column("responses", responses)
    ds = ds.filter(
        lambda batch: [True if response else False for response in batch["responses"]],
        batched=True,
        desc="Removing empty responses"
    )

    print(ds)
    print("Completions done!")


if __name__ == "__main__":
    asyncio.run(main())
