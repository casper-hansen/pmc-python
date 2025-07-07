# pip install vllm datasets
# vllm serve Qwen/Qwen3-Embedding-8B --task embed --disable-log-requests
import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENT_REQUESTS = 50
MAX_LENGTH = 32768

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
)


async def embed_text(text: str, sem: asyncio.Semaphore):
    """Create an embedding for a single text item, honoring the semaphore."""
    async with sem:
        response = await client.embeddings.create(
            input=[text],
            model="Qwen/Qwen3-Embedding-8B",
        )
        return response.data[0].embedding


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
        desc="Removing samples that are too long"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Generate embeddings
    tasks = [asyncio.create_task(embed_text(item["text"], semaphore)) for item in ds]
    embeddings = await tqdm_asyncio.gather(*tasks)

    print("Embeddings done!")


if __name__ == "__main__":
    asyncio.run(main())
