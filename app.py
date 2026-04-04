import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SensoEval", layout="wide")
st.title("Senso evaluation results")

# Load results
results_path = Path("data/results.json")

if not results_path.exists():
    st.warning("No results found. Run `python main.py` first to generate results.")
    st.stop()

with open(results_path) as f:
    results = json.load(f)

# Calculate metrics
total = len(results)
failed_count = sum(1 for r in results if r.get("issues"))
quality_scores = [r["quality_score"] for r in results if r["quality_score"] is not None]
consistency_scores = [r["consistency_score"] for r in results if r["consistency_score"] is not None]

avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

# Overview section
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Questions", total)
col2.metric("Avg Quality Score", f"{avg_quality:.2f}")
col3.metric("Avg Consistency", f"{avg_consistency:.2f}")
col4.metric("Failed Queries", failed_count)

st.divider()

# Results table
st.header("Results")
table_data = []
for r in results:
    answer_short = (r.get("answer", "")[:50] + "...") if r.get("answer") else "N/A"
    issues_str = ", ".join(r.get("issues", [])) or "✓"
    table_data.append({
        "ID": r["id"],
        "Question": r["question"][:40] + "...",
        "Answer": answer_short,
        "Quality": f"{r['quality_score']:.2f}" if r['quality_score'] is not None else "N/A",
        "Consistency": f"{r['consistency_score']:.2f}" if r['consistency_score'] is not None else "N/A",
        "Status": issues_str
    })

st.dataframe(table_data, use_container_width=True)

st.divider()

# Failure cases
st.header("Failure Cases")
failures = [r for r in results if r.get("issues")]

if not failures:
    st.success("✨ No failures detected!")
else:
    for f in failures:
        with st.expander(f"Q{f['id']}: {f['question'][:50]}... (Issues: {', '.join(f['issues'])})"):
            st.write(f"**Question:** {f['question']}")
            st.write(f"**Expected:** {f.get('expected_answer', 'N/A')}")
            st.write(f"**Answer:** {f.get('answer', 'N/A')}")
            st.write(f"**Quality Score:** {f['quality_score']}")
            st.write(f"**Consistency Score:** {f['consistency_score']}")
            if f.get('sources'):
                st.write(f"**Sources:** {f['sources']}")

st.divider()
st.caption("SensoEval - Simple evaluation prototype")
