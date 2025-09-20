# Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon

This repository contains the code and data for the research paper:

**"Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon"**

**Authors:** Nurit Cohen-Inger, Yehonatan Elisha, Bracha Shapira, Lior Rokach, Seffi Cohen

## Abstract

Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings, indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.

## Repository Structure

```
├── README.md                 # This file
├── data/                    # Dataset files and processed data
│   ├── rephrased_MMLU_Claude_0.5.csv
│   └── updated_mmlu.csv
├── scripts/                 # Python scripts for evaluation
│   ├── GPQA_eval_csv.py
│   └── mmlu_eval_csv.py
├── notebooks/               # Jupyter notebooks for analysis
│   ├── CBOD_rephrase_compared.ipynb
│   └── MMLU_rephrasing_Claude.ipynb
├── docs/                   # Documentation and papers
│   └── 2502.07445v2.pdf
└── results/                # Output results and figures
```

## Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed with the required packages.

#### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp config_template.env .env
   ```
   
   Then edit `.env` and add your API keys:
   - `ANTHROPIC_API_KEY`: Your Anthropic Claude API key
   - `OPENAI_API_KEY`: Your OpenAI API key (if needed)

### Running the Code

1. **MMLU Evaluation:**
   ```bash
   python scripts/mmlu_eval_csv.py
   ```

2. **GPQA Evaluation:**
   ```bash
   python scripts/GPQA_eval_csv.py
   ```

3. **Analysis Notebooks:**
   ```bash
   jupyter notebook notebooks/CBOD_rephrase_compared.ipynb
   jupyter notebook notebooks/MMLU_rephrasing_Claude.ipynb
   ```

## Data

The repository includes:
- `data/rephrased_MMLU_Claude_0.5.csv`: Rephrased MMLU dataset using Claude
- `data/updated_mmlu.csv`: Updated MMLU dataset

## Methods

This research introduces novel evaluation methodologies that account for the adaptive nature of LLMs, similar to how chameleons adapt to their environment.

## Results

Key findings and results can be found in:
- Analysis notebooks in the `notebooks/` directory
- Generated figures and tables in the `results/` directory

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{cohen2025forget,
      title={Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon}, 
      author={Nurit Cohen-Inger and Yehonatan Elisha and Bracha Shapira and Lior Rokach and Seffi Cohen},
      year={2025},
      eprint={2502.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```