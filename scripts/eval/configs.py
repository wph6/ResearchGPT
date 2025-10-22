# ================================================================
# 📂 File Path: opencompass/configs/datasets/research4k/research4k.py
# ---------------------------------------------------------------

from mmengine.config import read_base
from opencompass.models.openai_api import OpenAISDK

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess,research4k_llmjudge_postprocess
from opencompass.datasets import CustomDataset

from opencompass.datasets import Research4kDataset
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

judge_cfg = dict(
    abbr='gpt-4.1',               
    type=OpenAISDK,                     
    path='gpt-4.1',                     
    key='sk-xxxxxx',              # Replace with your API key
    openai_api_base=[
        '', 
    ],
    meta_template=api_meta_template,
    batch_size=128,
    temperature=0,
    max_out_len=2048,
    max_seq_len=4096,
    query_per_second=4,      
    retry=3,                        
)

reader_cfg = dict(input_columns=['question'], output_column='answer')

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Question: {question}?\nAnswer:'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


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

# Evaluation configuration using LLM as judge
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=Research4kDataset,
            path='<DATA_DIR>/research4k_test.json',
            reader_cfg=reader_cfg
        ),
        judge_cfg=judge_cfg,
        dict_postprocessor=dict(type = research4k_llmjudge_postprocess),
    ),
)

# Dataset configuration

research4k_datasets = [
    dict(
        type=Research4kDataset,
        path='<DATA_DIR>/research4k_test.json',
        reader_cfg=reader_cfg,
        infer_cfg = infer_cfg,
        eval_cfg = eval_cfg
    )
]