# ================================================================
# 📂 File Path: opencompass/datasets/research4k.py
# ---------------------------------------------------------------

import json
from datasets import Dataset
from opencompass.datasets import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from collections import defaultdict
import numpy as np
from opencompass.utils import get_logger
import re

# ================================================================
# 📑 Label order (research reasoning categories)
# ================================================================
LABEL_ORDER = [
    "research domain",
    "previous methods",
    "existing challenges",
    "motivation",
    "findings/assumptions",
    "methods",
    "experimental settings",
    "experimental results"
]

@LOAD_DATASET.register_module()
class Research4kDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                question = d['instruction'] + ('\n' + d['input'] if d['input'] else '')
                answer = d['output']
                label = d['label']
                dataset.append({
                    'question': question,
                    'answer': answer,
                    'label': label
                })
                # dataset.append({'question': question, 'answer': answer})
        return Dataset.from_list(dataset)

def _parse_research4k_score(raw: str):
    try:
        match = re.search(r'\b(?:10|[0-9])\b', raw.strip())
        if match:
            score = int(match.group(0))
            return {'score': score}
    except Exception as e:
        get_logger().warning(f"Failed to parse score: {e}")
    return None

def research4k_llmjudge_postprocess(
    output: dict,
    output_path: str,
    dataset: Dataset,
) -> dict:
    original_dataset = dataset.reader.dataset['test']

    scores = []
    details = {}
    per_label_scores = defaultdict(list)

    for k, v in output.items():
        idx = int(k) 
        raw = v['prediction']
        score_dict = _parse_research4k_score(raw)
        if score_dict is None:
            continue
        score = score_dict['score']       
        scores.append(score)

        details[k] = {
            'prediction': raw,
            'gold': v.get('gold', ''),
            'parsed_score': score_dict
        }
        sample = original_dataset[idx]
        label = sample.get("label", "unlabeled")
        per_label_scores[label].append(score)

    overall_avg = float(np.mean(scores)) * 10
    per_label_avg = {lab: float(np.mean(vals)) * 10 for lab, vals in per_label_scores.items()}
    per_label_count = {lab: float(len(vals)) for lab, vals in per_label_scores.items()}
    flat_metrics = {}

    for idx, lab in enumerate(LABEL_ORDER, start=1):
        if lab in per_label_avg:
            safe = lab.replace(' ', '_').replace('/', '-')
            key_avg   = f"R4K-{idx:02d}-{safe}"
            flat_metrics[key_avg]   = per_label_avg[lab]

    # =========================
    # Final output dictionary
    # =========================   
    agg = {
        'score_avg': overall_avg,
        'count': len(scores),
        'details': details,
        **flat_metrics,
    }

    return agg