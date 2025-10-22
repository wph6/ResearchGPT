from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.research4k.research4k import \
        research4k_datasets
    from opencompass.configs.models.gpt_5 import \
        models as gpt_5
datasets = research4k_datasets

models = gpt_5

# Output directory
work_dir = 'outputs/gpt_5'
