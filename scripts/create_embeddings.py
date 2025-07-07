# pip install vllm
# vllm serve Qwen/Qwen3-Embedding-8B --task embed
from openai import OpenAI
from datasets import load_dataset

ds = load_dataset(
    "casperhansen/pmc-oa-markdown",
    split="train",
    num_proc=8,
)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
)

response = client.embeddings.create(
    input=[ds[0]["text"]],
    model="Qwen/Qwen3-Embedding-8B"
)

print(response.data[0].embedding)
