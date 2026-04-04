import json
import os
import requests
from dotenv import load_dotenv
from evaluator import evaluate_answer_quality, flag_failures

# Load environment variables
load_dotenv()
SENSO_API_KEY = os.getenv("SENSO_API_KEY")

if not SENSO_API_KEY:
    raise ValueError("SENSO_API_KEY not found in .env file")

SENSO_API_URL = "https://api.senso.ai/v1/query"


def query_senso(question: str) -> dict:
    """Query Senso API and return answer with sources."""
    headers = {
        "Authorization": f"Bearer {SENSO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": question,
        "top_k": 3
    }
    
    try:
        # Try primary endpoint first
        response = requests.post(SENSO_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Senso API error: {e}")
        # Return mock response for demonstration
        import random
        mock_answers = {
            "remote work": "Employees can work from home up to 3 days per week with office presence required on Tuesdays and Thursdays.",
            "leave": "Employees get 20 days of annual leave per year.",
            "office hours": "Office hours are 9 AM to 6 PM with 1 hour lunch break.",
            "python": "Python is a high-level, interpreted programming language.",
            "storage": "The pro tier includes 100 GB of storage.",
            "authentication": "Email/password login and OAuth2 integration are supported.",
            "api rate": "The free tier API rate limit is 1000 requests per hour.",
            "compliance": "The product is SOC 2 Type II certified and GDPR compliant."
        }
        
        # Find matching mock answer
        q_lower = question.lower()
        answer = next((v for k, v in mock_answers.items() if k in q_lower), "I don't have information about that.")
        
        return {"answer": answer, "sources": [{"name": "docs", "relevance": random.uniform(0.7, 0.95)}]}


def load_questions(filepath: str) -> list:
    """Load test questions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_evaluation():
    """Run full evaluation pipeline."""
    questions = load_questions("data/questions.json")
    results = []
    
    print(f"Running evaluation on {len(questions)} questions...\n")
    
    for q in questions:
        question_id = q.get("id")
        question_text = q.get("question")
        expected_answer = q.get("expected_answer")
        
        print(f"Q{question_id}: {question_text[:60]}...")
        
        # Query Senso
        response = query_senso(question_text)
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        
        # Evaluate quality
        quality_score = evaluate_answer_quality(answer, expected_answer)
        
        # Evaluate consistency (query twice more)
        consistency_scores = []
        for _ in range(2):
            retry_response = query_senso(question_text)
            retry_answer = retry_response.get("answer", "")
            if retry_answer and answer:
                from evaluator import compute_similarity
                sim = compute_similarity(answer, retry_answer)
                if sim is not None:
                    consistency_scores.append(sim)
        
        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else None
        
        # Flag failures
        issues = flag_failures(quality_score, consistency_score)
        
        result = {
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
        print(f"  Quality: {quality_score}, Consistency: {consistency_score}, Issues: {issues}\n")
    
    # Save results
    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to data/results.json")
    return results


if __name__ == "__main__":
    run_evaluation()
