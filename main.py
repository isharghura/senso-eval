import json
import os
import requests
from dotenv import load_dotenv
from evaluator import evaluate_answer_quality, flag_failures
from anthropic import Anthropic
from openai import OpenAI

# Load environment variables
load_dotenv()
SENSO_API_KEY = os.getenv("SENSO_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"API Keys loaded: Senso={bool(SENSO_API_KEY)}, Claude={bool(ANTHROPIC_API_KEY)}, OpenAI={bool(OPENAI_API_KEY)}")

if not SENSO_API_KEY:
    raise ValueError("SENSO_API_KEY not found in .env file")

SENSO_API_URL = "https://apiv2.senso.ai/api/v1/org/search"

# Load config
with open("config.json") as f:
    config = json.load(f)


def load_context() -> str:
    docs = []
    doc_files = [
        "data/docs/company_policy.md",
        "data/docs/product_features.md", 
        "data/docs/python_intro.txt"
    ]
    for file_path in doc_files:
        try:
            with open(file_path, 'r') as f:
                docs.append(f.read())
        except FileNotFoundError:
            continue
    return "\n\n".join(docs)


CONTEXT = load_context()


def query_senso(question: str) -> dict:
    headers = {
        "X-API-Key": SENSO_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": question
    }
    
    try:
        # Try primary endpoint first
        response = requests.post(SENSO_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Senso API error: {e}")
        raise ValueError(f"Failed to query Senso API: {e}")


def query_claude(question: str) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        system_prompt = f"You are a helpful assistant. Answer questions based on the following context:\n\n{CONTEXT}"
        print(f"Claude system prompt length: {len(system_prompt)}")
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        answer = message.content[0].text
        print(f"Claude raw answer: '{answer}'")
        return answer
    except Exception as e:
        print(f"Claude API error: {e}")
        return ""


def query_openai(question: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        system_prompt = f"You are a helpful assistant. Answer questions based on the following context:\n\n{CONTEXT}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=1000,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ""


def load_questions(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def run_evaluation():
    questions = load_questions("data/questions.json")
    results = []
    
    models = [
        ("senso", query_senso),
        ("claude", query_claude),
        ("openai", query_openai)
    ]
    
    # Filter enabled models
    enabled_models = [(name, fn) for name, fn in models if config["models"].get(name, False)]
    
    print(f"Running evaluation on {len(questions)} questions for {len(enabled_models)} models...\n")
    
    for model_name, query_fn in enabled_models:
        print(f"Evaluating {model_name}...")
        for q in questions:
            question_id = q.get("id")
            question_text = q.get("question")
            expected_answer = q.get("expected_answer")
            
            print(f"  Q{question_id}: {question_text[:60]}...")
            
            # Query model
            if model_name == "senso":
                response = query_fn(question_text)
                answer = response.get("answer", "")
                sources = response.get("sources", [])
            else:
                answer = query_fn(question_text)
                sources = []
            
            print(f"    Answer: {answer[:100]}...")
            
            # Evaluate quality
            quality_score = evaluate_answer_quality(answer, expected_answer)
            
            # Evaluate consistency (query twice more)
            consistency_scores = []
            for _ in range(2):
                if model_name == "senso":
                    retry_response = query_fn(question_text)
                    retry_answer = retry_response.get("answer", "")
                else:
                    retry_answer = query_fn(question_text)
                if retry_answer and answer:
                    from evaluator import compute_similarity
                    sim = compute_similarity(answer, retry_answer)
                    if sim is not None:
                        consistency_scores.append(sim)
            
            consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else None
            
            # Flag failures
            issues = flag_failures(quality_score, consistency_score)
            
            result = {
                "model": model_name,
                "id": question_id,
                "question": question_text,
                "answer": answer,
                "expected_answer": expected_answer,
                "sources": sources,
                "quality_score": quality_score,
                "consistency_score": consistency_score,
                "issues": issues
            }
            results.append(result)
            print(f"    Quality: {quality_score}, Consistency: {consistency_score}, Issues: {issues}")
        print()
    
    # Save results
    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to data/results.json")
    
    # Print comparison summary
    print_comparison(results)
    
    return results


def print_comparison(results):
    from collections import defaultdict
    model_stats = defaultdict(lambda: {"quality": [], "consistency": [], "issues": 0})
    
    for r in results:
        model = r["model"]
        if r["quality_score"] is not None:
            model_stats[model]["quality"].append(r["quality_score"])
        if r["consistency_score"] is not None:
            model_stats[model]["consistency"].append(r["consistency_score"])
        model_stats[model]["issues"] += len(r["issues"])
    
    print("Comparison Summary:")
    for model, stats in model_stats.items():
        avg_quality = sum(stats["quality"]) / len(stats["quality"]) if stats["quality"] else 0
        avg_consistency = sum(stats["consistency"]) / len(stats["consistency"]) if stats["consistency"] else 0
        print(f"  {model}: Avg Quality={avg_quality:.2f}, Avg Consistency={avg_consistency:.2f}, Total Issues={stats['issues']}")


if __name__ == "__main__":
    run_evaluation()
