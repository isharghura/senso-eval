# SensoEval

Evaluate Senso, Claude, and OpenAI on question answering over company documents.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set API keys in `.env`:
   ```
   SENSO_API_KEY=your_senso_key
   ANTHROPIC_API_KEY=your_claude_key
   OPENAI_API_KEY=your_openai_key
   ```

3. Configure models in `config.json`:
   ```json
   {
     "models": {
       "senso": true,
       "claude": true,
       "openai": false
     }
   }
   ```

## Usage

Run evaluation:
```bash
python main.py
```

This queries enabled models for each question in `data/questions.json` and saves results to `data/results.json`.

## Results

Results include quality scores (similarity to expected answers) and consistency scores (similarity between retries). Comparison summary shows average performance per model.
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