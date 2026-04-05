# Model Evaluation Pipeline

Simple evaluation framework for comparing LLMs (Senso, Claude, OpenAI) on a set of questions using quality + consistency metrics.

## What it does

- Runs questions from `data/questions.json` against multiple models
- Scores answers using:
  - **LLM judge** (Gemini / Claude / OpenAI fallback)
  - **Embedding similarity** (MiniLM)
- Measures:
  - **Quality** (how correct the answer is)
  - **Consistency** (does the model give similar answers repeatedly)
- Flags failures:
  - `low_quality`
  - `low_consistency`
- Saves results to `data/results.json`
- Prints a summary comparison

---

## Setup

### 1. Install deps

```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python main.py
```