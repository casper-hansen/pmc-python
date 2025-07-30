"""
In this script, I perform synthetic data generation and extensive filtering.

3 modes of filtering:
1. Correctness: Answers correctly with context? Answers incorrectly without context?
2. Diversity: Deduplicate too similar questions and answers
3. Quality: Scores well on rubric
"""

# uv venv
# source .venv/bin/activate
import asyncio
import copy
import hashlib
import json
import os
import random
from tqdm import tqdm
from typing import List, Dict, Literal, Callable, Awaitable, Union
from pydantic import BaseModel, Field, StringConstraints
from openai import AsyncOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ParsedChatCompletion, ChatCompletion
from datasets import load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from semhash import SemHash

MAX_CONCURRENT_REQUESTS = 100
LOADED_BATCH_SIZE = 1000
SYNTH_MAX_LENGTH = 32768
RUBRIC_MAX_LENGTH = 16384
MAX_SAMPLE_LENGTH = 163840 - SYNTH_MAX_LENGTH - RUBRIC_MAX_LENGTH
CACHE_PATH = "data/qa_cache.jsonl"
MODEL = "deepseek-ai/DeepSeek-R1-0528"
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

RUBRIC_PROMPT = """\
Below are 5 different metrics that I want you to judge given a triplet of [question], [answer], [context].

Reasoning: Explain why metric scored between 0 to 5. Do not comment on any background to the problem, do not attempt to solve the problem, and do not argue for any answer different than the provided answer.

Metrics:
- Difficulty (high school | graduate | phd): cognitive depth required to understand and address the [question].
- Askability (0-5):  biological soundness and contextual fit—are the concepts plausibly linked within current biomedical knowledge and do they arise naturally from the provided [context]?
- Synthesizability (0-5): extent to which the [answer] must integrate information from multiple full-length biomedical sources in [context].
- Abstractiveness (0-5): degree of conceptual abstraction beyond the literal [context]; 0 = verbatim extraction.
- Conflict Synthesis (0-5): presence of contradictory evidence demanding nuanced reconciliation across [context].

[question]: {question}

[answer]: {answer}

[context]:

{context_text}
"""

ANSWER_PROMPT = """\
You are a biomedical expert. Below, You must attempt to answer the question below correctly.

Question: {question}
"""

ANSWER_PROMPT_WITH_CONTEXT = (
    ANSWER_PROMPT
    + """
## Context

{texts}
"""
)

JUDGE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""


class ExtractedQuestionAnswer(BaseModel):
    question: str = StringConstraints(min_length=1)
    answer: str = StringConstraints(min_length=1)
    strict: Literal[True]


class RubricFilter(BaseModel):
    difficulty: Literal["high school", "graduate", "phd"]
    askability: int = Field(ge=0, le=5)
    synthesizability: int = Field(ge=0, le=5)
    abstractiveness: int = Field(ge=0, le=5)
    conflict_synthesis: int = Field(ge=0, le=5)
    strict: Literal[True]


class JudgeTripletFilter(BaseModel):
    extracted_final_answer: str = StringConstraints(min_length=1)
    reasoning: str = StringConstraints(min_length=1)
    correct: Literal["yes", "no"]
    strict: Literal[True]


class CacheRecord(BaseModel):
    sample_id: int | None = None
    qa: ExtractedQuestionAnswer | None
    rubric: RubricFilter | None
    judge_with_context: JudgeTripletFilter | None
    judge_without_context: JudgeTripletFilter | None


EMPTY = CacheRecord(
    sample_id=None, qa=None, rubric=None, judge_with_context=None, judge_without_context=None
)

client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "test-key"),
)

cache: dict[str, CacheRecord] = {}

if os.path.exists(CACHE_PATH):
    ds_cache = load_dataset(
        "json",
        data_files=[CACHE_PATH],
        split="train",
        streaming=True,
    )
    with open(CACHE_PATH, "r") as fh:
        for row in ds_cache:
            try:
                cache[row["hash"]] = CacheRecord(**row["response"])
            except (KeyError, TypeError, ValueError):
                # if the row is malformed or missing keys, skip it
                continue
    print(f"Loaded {len(cache)} cached completions from {CACHE_PATH}.")


async def async_backoff(
    func: Callable[..., Awaitable],
    *args,
    max_retries: int = 6,
    initial_delay: float = 10.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.2,
    **kwargs,
) -> Union[ParsedChatCompletion, ChatCompletion]:
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as err:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0, delay * jitter)
            print(
                f"[Rate-limit]: {attempt+1}/{max_retries} "
                f"in {sleep_for:.1f}s ({err.message})"
            )
            await asyncio.sleep(sleep_for)
            delay *= backoff_factor

async def _get_record(texts) -> CacheRecord:
    try:
        qa_resp = await async_backoff(
            client.chat.completions.parse,
            messages=[
                {
                    "role": "user",
                    "content": SYNTH_PROMPT.format(texts=_join(texts)),
                }
            ],
            model=MODEL,
            max_completion_tokens=SYNTH_MAX_LENGTH,
            response_format=ExtractedQuestionAnswer,
        )
        qa = qa_resp.choices[0].message.parsed

        if qa is None:
            return EMPTY

        rubric_resp = await async_backoff(
            client.chat.completions.parse,
            messages=[
                {
                    "role": "user",
                    "content": RUBRIC_PROMPT.format(
                        question=qa.question,
                        answer=qa.answer,
                        context_text=_join(texts),
                    ),
                }
            ],
            model=MODEL,
            max_completion_tokens=RUBRIC_MAX_LENGTH,
            response_format=RubricFilter,
        )
        rubric = rubric_resp.choices[0].message.parsed

        if rubric is None:
            return EMPTY

        response_no_context = await async_backoff(
            client.chat.completions.create,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=qa.question),
                }
            ],
            model=MODEL,
        )

        judge_resp = await async_backoff(
            client.chat.completions.parse,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=qa.question,
                        response=response_no_context.choices[0].message.content,
                        correct_answer=qa.answer,
                    ),
                }
            ],
            model=MODEL,
            max_completion_tokens=32768,
            response_format=JudgeTripletFilter,
        )
        judgement = judge_resp.choices[0].message.parsed
        if judgement is None:
            return EMPTY

        response_with_context = await async_backoff(
            client.chat.completions.create,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT_WITH_CONTEXT.format(
                        question=qa.question,
                        texts=_join(texts),
                    ),
                }
            ],
            model=MODEL,
        )

        judge_resp_context = await async_backoff(
            client.chat.completions.parse,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=qa.question,
                        response=response_with_context.choices[0].message.content,
                        correct_answer=qa.answer,
                    ),
                }
            ],
            model=MODEL,
            max_completion_tokens=32768,
            response_format=JudgeTripletFilter,
        )
        judgement_context = judge_resp_context.choices[0].message.parsed
        if judgement_context is None:
            return EMPTY

    except BadRequestError as ex:
        print("Error:", ex)
        return EMPTY
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print("Exiting process due to unknown error...")
        exit(0)
    
    return CacheRecord(
        qa=qa,
        rubric=rubric,
        judge_with_context=judgement_context,
        judge_without_context=judgement,
    )


async def create_completion(
    sample_id: int, texts: List[str], sem: asyncio.Semaphore, lock: asyncio.Lock
) -> CacheRecord:
    """Return completion for *text*, using cache when available.

    If the completion is not cached, it is requested from the model, then stored
    immediately so progress persists across crashes/restarts.
    """
    key = hashlib.md5(
        (
            json.dumps(texts, ensure_ascii=False)
            + SYNTH_PROMPT
            + RUBRIC_PROMPT
            + ANSWER_PROMPT
            + ANSWER_PROMPT_WITH_CONTEXT
        ).encode("utf-8")
    ).hexdigest()

    # Reuse from cache if we already have it.
    if key in cache:
        record = cache[key]
    else:
        async with sem:
            record: CacheRecord = await _get_record(texts)

        record = copy.deepcopy(record)
        record.sample_id = sample_id

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

    # filter for correctness and quality
    is_acceptable_record = (
        record.qa is not None
        and record.rubric is not None
        and record.judge_with_context is not None
        and record.judge_without_context is not None
        and record.judge_with_context.correct == "yes"
        and record.judge_without_context.correct == "no"
        and record.rubric.difficulty in ["graduate", "phd"]
        and record.rubric.askability >= 4
        and record.rubric.synthesizability >= 4
        and record.rubric.abstractiveness >= 4
        and record.rubric.conflict_synthesis >= 4
    )

    if not is_acceptable_record:
        record = copy.deepcopy(EMPTY)
        record.sample_id = sample_id
        return record
    
    return record


def _join(texts):
    return "\n\n".join(
        f"# Article {i+1}/{len(texts)}:\n\n{text}" for i, text in enumerate(texts)
    )


async def main():
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown-clustering",
        split="train",
        num_proc=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
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
    for batch_idx in tqdm(
        range(num_batches), desc=f"Batches (samples={num_rows})", unit="batch"
    ):
        start = batch_idx * LOADED_BATCH_SIZE
        end = min(start + LOADED_BATCH_SIZE, num_rows)
        batch = ds.select(range(start, end))

        # Spawn completion tasks for this batch only.
        tasks = [
            asyncio.create_task(create_completion(start + local_idx, texts, semaphore, CACHE_WRITE_LOCK))
            for local_idx, texts in enumerate(batch["texts"])
        ]

        # Inner progress bar for the current batch.
        batch_responses = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Completions (batch {batch_idx + 1}/{num_batches})",
            leave=False,
        )
        responses.extend(batch_responses)

    # Collect indices that have both qa and rubric
    keep_indices = [
        r.sample_id
        for r in responses
        if r.qa is not None and r.rubric is not None
    ]

    ds = ds.select(keep_indices)

    ds = ds.add_column("question", [responses[idx].qa.question for idx in keep_indices])
    ds = ds.add_column("answer", [responses[idx].qa.answer for idx in keep_indices])
    ds = ds.remove_columns(["embeds", "avg_similarity"])
    ds = ds.rename_column("texts", "context")

    # deduplication to increase diversity
    records = [{"id": i, "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(len(ds))]
    semhash   = SemHash.from_records(records, columns=["question", "answer"])
    dedup_res = semhash.self_deduplicate(threshold=0.9)
    keep_ids = [r["id"] for r in dedup_res.selected]
    ds = ds.select(keep_ids)

    print(f"Kept {len(keep_ids):,} / {len(records):,} after SemHash")
    print(ds)
    print("Completions done!")

    # save dataset and cache on hub
    DatasetDict({"train": ds}).push_to_hub(
        "casperhansen/pmc-oa-markdown-qa"
    )
    load_dataset("json", data_files=[CACHE_PATH]).push_to_hub(
        "casperhansen/pmc-oa-markdown-qa-cache"
    )


if __name__ == "__main__":
    asyncio.run(main())
