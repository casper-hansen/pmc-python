# uv venv
# source .venv/bin/activate
import asyncio
import hashlib
import json
import os
from tqdm import tqdm
from typing import List, Dict, Literal
from pydantic import BaseModel, Field
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

SYNTH_PROMPT = """\
You are an expert biomedical researcher tasked with synthesizing knowledge from multiple full-length biomedical documents that have been grouped together because they share extremely high topical similarity. Your goal is to generate a single, graduate-level question and its corresponding answer that can **only** be resolved by integrating information scattered across **all** provided texts. The question must be so demanding that a reader who has access to only one of the texts, or even most of them, would find it nearly impossible to answer correctly.

## Core Requirements

1. **Graduate-Level Difficulty**
    - Questions must probe deep conceptual relationships, methodological subtleties, or conflicting evidence that requires advanced domain knowledge to reconcile.
    - Avoid surface-level facts or summaries that could be extracted from any single section.

2. **Cross-Text Integration**
    - The answer must weave together findings, data, or arguments that appear in **different** parts of **different** texts.
    - No single text should contain enough information to answer the question fully.

3. **Neutrality of Source**
    - Never mention “the study,” “the paper,” “the article,” or any reference to the documents themselves.
    - Speak as if the knowledge is universal, not tied to specific publications.

4. **Answerability**
    - The question must be unambiguously answerable if and only if the entire collection of texts has been read and synthesized.
    - Provide a concise yet complete answer that demonstrates the synthesis.

## Example Style (Do Not Copy)

- **Poor**: “What is the sample size in the first experiment?” (single-text, trivial)
- **Good**: “How do the differential expression patterns of lncRNA X in hypoxic endothelial cells reconcile the apparently contradictory roles attributed to it in angiogenesis, and what post-transcriptional mechanisms are proposed to explain this discrepancy?” (requires cross-text synthesis of expression data, functional assays, and mechanistic models)

Generate exactly one such question-answer pair.

## Context

{texts}
"""

RUBRIC_PROMPT = """
Below are 5 different metrics that I want you to judge given a triplet of [question], [answer], [context].

Reasoning: Explain why metric scored between 0 to 5. Do not comment on any background to the problem, do not attempt to solve the problem, and do not argue for any answer different than the provided answer.

Metrics:
- Difficulty (high school | graduate | phd): cognitive depth required to understand and address the [question].
- Askability (0-5): biological plausibility and naturalness—would a domain expert actually pose this [question]?
- Synthesizability (0-5): extent to which the [answer] must integrate information from multiple full-length biomedical sources in [context].
- Abstractiveness (0-5): degree of conceptual abstraction beyond the literal [context]; 0 = verbatim extraction.
- Conflict Synthesis (0-5): presence of contradictory evidence demanding nuanced reconciliation across [context].

[question]: {question}

[answer]: {answer}

[context]:

{context_text}
"""

class ExtractedQuestionAnswer(BaseModel):
    question: str
    answer: str
    strict: Literal[True]

class MetricScore(BaseModel):
    reasoning: str
    score: int = Field(ge=0, le=5)

class RubricFilter(BaseModel):
    difficulty: Literal["high school", "graduate", "phd"]
    askability: MetricScore
    synthesizability: MetricScore
    abstractiveness: MetricScore
    conflict_synthesis: MetricScore
    strict: Literal[True]

class JudgeTripletFilter(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    strict: Literal[True]

class CacheRecord(BaseModel):
    qa: ExtractedQuestionAnswer | None
    rubric: RubricFilter | None

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
                cache[record["hash"]] = ExtractedQuestionAnswer(**record["response"])
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., partial writes)
                continue
    print(f"Loaded {len(cache)} cached completions from {CACHE_PATH}.")


async def create_completion(texts: List[str], sem: asyncio.Semaphore, lock: asyncio.Lock) -> CacheRecord:
    """Return completion for *text*, using cache when available.

    If the completion is not cached, it is requested from the model, then stored
    immediately so progress persists across crashes/restarts.
    """
    key = hashlib.md5(
        json.dumps(texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    # Reuse from cache if we already have it.
    if key in cache:
        cached: CacheRecord = cache[key]
        return cached

    # Get chain of model completions
    qa = None
    rubric = None
    async with sem:
        try:
            qa_resp = await client.chat.completions.parse(
                messages=[{
                    "role": "system",
                    "content": SYNTH_PROMPT.format(texts=_join(texts)),
                }],
                model=MODEL,
                max_completion_tokens=4096,
                response_format=ExtractedQuestionAnswer,
            )
            qa: ExtractedQuestionAnswer = qa_resp.choices[0].message.parsed

            rubric_resp = await client.chat.completions.parse(
                messages=[{
                    "role": "system",
                    "content": RUBRIC_PROMPT.format(
                        question=qa.question,
                        answer=qa.answer,
                        context_text=_join(texts),
                    ),
                }],
                model=MODEL,
                max_completion_tokens=1024,
                response_format=RubricFilter,
            )
            rubric: RubricFilter = rubric_resp.choices[0].message.parsed
        except BadRequestError as ex:
            print("RUBRIC error:", ex)
    
    record = CacheRecord(qa=qa, rubric=rubric)

    async with lock:
        cache[key] = record
        with open(CACHE_PATH, "a") as fh:
            fh.write(
                json.dumps(
                    {"hash": key, "response": record.model_dump()},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return record


def _join(texts):
    return "\n\n".join(f"Article {i+1}:\n\n{text}" for i, text in enumerate(texts))


async def main():
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown-clustering",
        split="train",
        num_proc=8,
    ).take(1)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    ds = ds.filter(
        lambda batch: [
            n_tokens + 20 <= MAX_SAMPLE_LENGTH
            for n_tokens in tokenizer(
                [SYNTH_PROMPT.format(texts=_join(texts)) for texts in batch["texts"]],
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

    responses: List[CacheRecord] = []

    # Outer progress bar for batches.
    for batch_idx in tqdm(range(num_batches), desc=f"Batches (samples={num_rows})", unit="batch"):
        start = batch_idx * LOADED_BATCH_SIZE
        end = min(start + LOADED_BATCH_SIZE, num_rows)
        batch = ds.select(range(start, end))

        # Spawn completion tasks for this batch only.
        tasks = [
            asyncio.create_task(
                create_completion(texts, semaphore, CACHE_WRITE_LOCK)
            )
            for texts in batch["texts"]
        ]

        # Inner progress bar for the current batch.
        batch_responses = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Completions (batch {batch_idx + 1}/{num_batches})",
            leave=False,
        )
        responses.extend(batch_responses)
    
    ds = ds.add_column("question", [r.qa.question for r in responses if r.qa])
    ds = ds.add_column("answer", [r.qa.answer for r in responses if r.qa])
    ds = ds.remove_columns(["embeds", "avg_similarity"])
    print(ds)
    print("Completions done!")


if __name__ == "__main__":
    asyncio.run(main())
