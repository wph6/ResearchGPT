# 🧠 ResearchGPT: Benchmarking and Training LLMs for End-to-End Computer Science Research Workflows


## ✨ Overview
![CS-4k Overview](figures/intro.png)

As large language models (LLMs) advance, the ultimate vision for their role in science is emerging: we could build an AI collaborator to effectively assist human beings throughout the entire scientific research process. We refer to this envisioned system as \emph{ResearchGPT}.
To move toward this goal, we present CS-54k, a high-quality corpus of computer science Q&A pairs derived from 14k CC-licensed papers through a scalable, paper-grounded pipeline combining RAG and multi-stage quality control.

From CS-54k, we derive two subsets:
- **CS-4k**: a benchmark for evaluating end-to-end research-assistant capabilities;
- **CS-50k**: a large-scale training dataset for domain-aligned model development.

Experiments show that even 7B-scale open models fine-tuned on CS-50k surpass larger proprietary systems (e.g., GPT-4.1, GPT-4o, Gemini 2.5 Pro).
This indicates that making AI models better research assistants relies more on domain-aligned training with high-quality data than on pretraining scale or general benchmark performance.


---
## 🧱 Dataset Construction
### Pipeline Overview
A scalable paper-grounded pipeline combining RAG with multi-stage quality control to ensure factual grounding and reproducibility.

![pipeline](figures/pipeline.png)
---
### Dataset Distributions
Distributions of topic category, difficulty level, and input length across the CS-54k corpus.

![detail](figures/detail.png)

### Topic Categories
The dataset organizes each Q&A pair into one of eight topic classes, reflecting distinct reasoning functions within scientific papers:
![category](figures/category.png)

## ⚙️Training

## 📈Evaluation

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📚 Citation
If you find ResearchGPT useful, please cite our paper:

