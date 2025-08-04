"""
Get started:
uv pip install tqdm pydantic openai[aiohttp] datasets
nohup uv run python -u scripts/eval_qa.py > eval.log 2>&1 &

The following environment variables are expected:
export OPENAI_BASE_URL_EVAL=xyz
export OPENAI_API_KEY_EVAL=123
export OPENAI_BASE_URL_JUDGE=xyz
export OPENAI_API_KEY_JUDGE=123
"""

# uv venv
# source .venv/bin/activate
import asyncio
import copy
import hashlib
import httpx
import json
import os
import random
from typing import List, Literal, Callable, Awaitable, Union, Tuple
from pydantic import BaseModel, StringConstraints
from openai import AsyncOpenAI, DefaultAioHttpClient, BadRequestError, LengthFinishReasonError
from openai.types.chat import ParsedChatCompletion, ChatCompletion
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENT_REQUESTS = 50
CACHE_PATH = "data/qa_cache_eval.jsonl"
EVAL_MODEL = "zai-org/GLM-4.5-FP8"
JUDGE_MODEL = "o4-mini"
CACHE_WRITE_LOCK = asyncio.Lock()

ANSWER_PROMPT = """\
You are a biomedical expert. Below, You must attempt to answer the question below with a correct conclusion.

Question: {question}
"""

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


class JudgeTripletFilter(BaseModel):
    extracted_final_answer: str = StringConstraints(min_length=1)
    reasoning: str = StringConstraints(min_length=1)
    correct: Literal["yes", "no"]
    strict: Literal[True]


class CacheRecord(BaseModel):
    sample_id: int | None = None
    judge: JudgeTripletFilter | None


EMPTY = CacheRecord(sample_id=None, judge=None)

client_eval = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL_EVAL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY_EVAL", "test-key"),
    http_client=DefaultAioHttpClient(timeout=httpx.Timeout(900))
)

client_judge = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL_JUDGE", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY_JUDGE", "test-key"),
    http_client=DefaultAioHttpClient(timeout=httpx.Timeout(900))
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
        except Exception as err:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0, delay * jitter)
            print(
                f"[Error]: {attempt+1}/{max_retries} "
                f"in {sleep_for:.1f}s ({err})"
            )
            await asyncio.sleep(sleep_for)
            delay *= backoff_factor

async def _get_record(question: str, answer: str) -> CacheRecord:
    try:
        if "deep-research" in EVAL_MODEL:
            response_no_context = await async_backoff(
                client_eval.responses.create,
                input=ANSWER_PROMPT.format(question=question),
                model=EVAL_MODEL,
                tools=[
                    {"type": "web_search_preview"},
                ]
            )

            generated_answer = response_no_context.output_text
        else:
            response_no_context = await async_backoff(
                client_eval.chat.completions.create,
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_PROMPT.format(question=question),
                    }
                ],
                model=EVAL_MODEL,
            )

            generated_answer = response_no_context.choices[0].message.content

        judge_resp = await async_backoff(
            client_judge.chat.completions.parse,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        response=generated_answer,
                        correct_answer=answer,
                    ),
                }
            ],
            model=JUDGE_MODEL,
            max_completion_tokens=32768,
            response_format=JudgeTripletFilter,
        )
        judgement = judge_resp.choices[0].message.parsed
        if judgement is None:
            return EMPTY

    except (BadRequestError, LengthFinishReasonError) as ex:
        print("Error:", ex)
        return EMPTY
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print("Exiting process due to unknown error...")
        exit(0)
    
    return CacheRecord(judge=judgement)


async def create_completion(
    sample_id: int, question: str, answer: str, sem: asyncio.Semaphore, lock: asyncio.Lock
) -> CacheRecord:
    """Return completion for *text*, using cache when available.

    If the completion is not cached, it is requested from the model, then stored
    immediately so progress persists across crashes/restarts.
    """
    key = hashlib.md5(
        (
            question
            + answer
            + ANSWER_PROMPT
            + JUDGE_PROMPT
        ).encode("utf-8")
    ).hexdigest()

    # Reuse from cache if we already have it.
    if key in cache:
        record = cache[key]
    else:
        async with sem:
            record: CacheRecord = await _get_record(question, answer)

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
    
    return record


def compute_accuracy(responses: List[CacheRecord]) -> Tuple[float, float, float]:
    correct = incorrect = 0
    for rec in responses:
        if rec.judge is None or rec.judge.correct == "no":
            incorrect += 1
        elif rec.judge.correct == "yes":
            correct += 1

    total = correct + incorrect
    if total == 0:
        return 0.0

    accuracy = correct / total

    return accuracy, correct, incorrect

async def main():
    ds = load_dataset(
        "casperhansen/pmc-oa-markdown-qa",
        split="test",
        num_proc=8,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Spawn completion tasks for this batch only.
    tasks = [
        asyncio.create_task(create_completion(index, row["question"], row["answer"], semaphore, CACHE_WRITE_LOCK))
        for index, row in enumerate(ds)
    ]

    # Inner progress bar for the current batch.
    responses: List[CacheRecord] = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Running evaluation",
    )

    accuracy, correct, incorrect = compute_accuracy(responses)

    print(f"** Accuracy: {accuracy*100:.2f}% ({correct}/{incorrect+correct} were correct)")


if __name__ == "__main__":
    asyncio.run(main())
