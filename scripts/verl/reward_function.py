# ================================================================
#  Reward Function for CS-50k in VERL
# ================================================================


from concurrent.futures import ThreadPoolExecutor
import os
from openai import OpenAI
import requests
import time
from loguru import logger as eval_logger
import re

NUM_SECONDS_TO_SLEEP = 5
MAX_WORKERS = 32

MODEL_VERSION = "Qwen/Qwen2.5-7B-Instruct"
API_URL = os.getenv("OPENAI_API_BASE", "http://localhost:30000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
client = OpenAI(base_url=API_URL, api_key=API_KEY)

# ------------------------------------------------
# 🧠 Grading Template
# ------------------------------------------------
GRADER_TEMPLATE = """
You are a grading expert, judge whether the final answers given by the candidates below are consistent with the reference answers, that is, whether the candidates answered correctly.
Provide a single **integer score from 0 to 10**.

1. Scoring Guide
- **9-10**: Fully correct, detailed, and clearly structured. The prediction matches the reference not only in meaning, but also in technical detail, terminology, and step-by-step structure. Only assign this score if the answer is nearly indistinguishable from the reference.
- **7-8**: Covers all key points and includes all key technical details, but may have slight simplifications, structural looseness, or phrasing issues. The prediction is still technically sound and semantically aligned.
- **5-6**: Covers nearly all major technical points with acceptable clarity and structure.The prediction may slightly oversimplify or omit minor supporting details, but the core reasoning is intact.
- **3-4**: Incomplete or partially correct. Misses one or more core concepts or steps, or is expressed in a vague, confusing, or disorganized way.
- **1-2**: Weak or flawed answer. Contains major factual errors, misunderstands the question, or mixes unrelated ideas. May reference a few relevant terms, but lacks meaningful explanation or structure.
- **0**: Fundamentally incorrect or completely off-topic. No meaningful alignment with the reference content, or content is nonsensical or missing.


2. Output Format  
Respond with a **single integer from 0 to 10**.  
Do not include any explanation or additional text.

3. Special Instructions
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- Do not re-answer the question yourself.
- Assign a high score only if the prediction matches the answer **semantically and technically**, considering variations in format.
- Deduct points for missing key technical details or excessive generalizations. Even if the tone is correct, factual or structural omissions should lead to a reduced score.
- Be strict with general or vague answers: If the prediction only provides a high-level overview but omits key technical details, steps, or quantitative findings present in the reference, score **no higher than 6**.
- Ignore minor differences in formatting, capitalization, or spacing.

Example Response:  
7

Now start your task. 

# Input

Question:  
{question}

Reference Answer:  
{answer}

Model Prediction:  
{prediction}

""".strip()

# ------------------------------------------------
# Query Local LLM Judge
# ------------------------------------------------
def get_chat_response(content: str, max_tokens: int, retries: int = 5):
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            if attempt == retries - 1:
                eval_logger.error(f"Failed to get response after {retries} attempts")
                return ""
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return ""

# ------------------------------------------------
#  Reward Normalization
# ------------------------------------------------
def compute_reward(response):
    """
    Parse integer score from model response and normalize to [0, 1].

    Args:
        response (str): Model output (expected 0–10)

    Returns:
        float: Normalized reward in [0.0, 1.0]
    """
    
    reward_score = 0.0
    try:
        match = re.search(r'\b(?:10|[0-9])\b', response.strip())
        if match:
            score = int(match.group(0))
            reward_score = score / 10.0  
    except Exception as e:
        print(e)
    return reward_score

# ------------------------------------------------
#  Single-Sample Evaluation
# ------------------------------------------------
def compute_score(data_source, solution_str, ground_truth, extra_info):

    problem = extra_info["question"]

    llm_judge_prompt = GRADER_TEMPLATE.format(question=problem, answer=ground_truth, prediction=solution_str)
    response = get_chat_response(llm_judge_prompt, max_tokens=20, retries=3)

    if response is not None:
        reward_score = compute_reward(response)
    else:
        reward_score = 0.0

    return reward_score

# ------------------------------------------------
#  Batch Evaluation (Parallel)
# ------------------------------------------------
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]

    return results
