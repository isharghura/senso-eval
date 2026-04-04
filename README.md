# SensoEval

Simple prototype to evaluate how well Senso answers questions over a set of documents.

## Features

- **Load Documents**: Sample documents in `data/docs/` (txt/markdown)
- **Test Questions**: ~10-15 questions in `data/questions.json` with optional expected answers
- **Query Senso**: Send each question to Senso API and store results
- **Evaluate Quality**: Compute answer similarity using sentence-transformers embeddings
- **Check Consistency**: Query same question twice, compare answers
- **Failure Detection**: Flag low quality or low consistency answers
- **Visualize Results**: Streamlit UI with overview metrics, results table, and failure cases

## Structure

```
.
├── main.py              # Run evaluation, query Senso, save results
├── evaluator.py         # Metrics: similarity, consistency, failure flags
├── app.py              # Streamlit UI
├── requirements.txt    # Dependencies
├── data/
│   ├── docs/           # Sample documents
│   ├── questions.json  # Test questions
│   └── results.json    # Evaluation results (generated)
└── .env                # SENSO_API_KEY
```

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Set API Key
Ensure `.env` contains your Senso API key:
```
SENSO_API_KEY=your_key_here
```

### 3. Run Evaluation
```bash
python3 main.py
```

This queries Senso for each question and saves results to `data/results.json`.

### 4. View Results
```bash
streamlit run app.py
```

Opens UI at `http://localhost:8501` showing:
- **Overview**: Average quality/consistency scores, failure count
- **Results Table**: All queries with scores and status
- **Failure Cases**: Expanded view of failed queries

## Metrics

- **Quality Score** (0-1): Similarity between answer and expected answer
- **Consistency Score** (0-1): Similarity between two responses to same question
- **Failure Flags**: `low_quality` (< 0.5) or `low_consistency` (< 0.7)

## Example

```python
from evaluator import compute_similarity, check_consistency

# Similarity between two texts
score = compute_similarity("Python is a language", "Python is a programming language")
# Returns: ~0.95

# Consistency check
def query_fn(q):
    return "Python is great"

consistency = check_consistency("What is Python?", query_fn)
# Returns: consistency score based on multiple queries
```

## Tech Stack

- **Python 3.8+**
- **requests**: HTTP calls
- **sentence-transformers**: Embeddings for similarity
- **streamlit**: Web UI
- **python-dotenv**: Load environment variables

## Notes

- Everything runs locally
- No database needed
- Mock Senso responses if API unavailable
- Simple, minimal UI focused on evaluation
