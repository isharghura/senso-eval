from sentence_transformers import SentenceTransformer, util
from typing import Optional, Callable, List
import numpy as np
import re
from typing import Optional, Callable


class EmbeddingCache:
    def __init__(self):
        self.cache = {}
    
    def get_or_compute(self, model, text):
        """Get cached embedding or compute if not present."""
        if text is None or text == "":
            return None
        
        key = hash(text)
        if key not in self.cache:
            self.cache[key] = model.encode(text)
        return self.cache[key]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


# Global embedding cache
_embedding_cache = EmbeddingCache()
_model = None


def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def compute_similarity(text1: str, text2: str) -> Optional[float]:
    if text1 is None or text2 is None:
        return None
    
    text1 = str(text1).strip()
    text2 = str(text2).strip()
    
    if not text1 or not text2:
        return None
    
    model = _get_model()
    
    # Get cached or compute embeddings
    embedding1 = _embedding_cache.get_or_compute(model, text1)
    embedding2 = _embedding_cache.get_or_compute(model, text2)
    
    if embedding1 is None or embedding2 is None:
        return None
    
    # Compute cosine similarity using sentence_transformers utility
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    
    # Convert to float and scale to 0-1 range
    score = float(similarity.item())
    return max(0.0, min(1.0, score))


import re
from typing import Optional, Callable

def evaluate_answer_quality(
    answer: str,
    expected_answer: Optional[str] = None,
    judge_query_fn: Optional[Callable[[str], str]] = None,
    question: Optional[str] = None,
) -> Optional[float]:
    if expected_answer is None:
        return None
    if not answer:
        return 0.0

    if judge_query_fn is None or question is None:
        # fallback
        if contains_expected(answer, expected_answer):
            return 0.8
        sim = compute_similarity(answer, expected_answer)
        return sim if sim is not None else 0.0

    # === LLM JUDGE ===
    judge_prompt = f"""You are a senior AI researcher grading an evaluation.
Compare the 'Model Answer' to the 'Reference Answer' for the question below.

Question: {question}
Reference Answer: {expected_answer}
Model Answer: {answer}

Grading Logic:
1. The Reference Answer is a high-level summary.
2. The Model Answer describes the technical cause (e.g., k-means, offline preprocessing, GPU inefficiency) of the symptoms in the Reference (e.g., slow indexing).
3. If the Model Answer is technically accurate and explains the "why" behind the Reference Answer, it should receive a high score (0.8 - 1.0).
4. Do NOT penalize the model for missing the specific name "TurboQuant" unless the question asked for a comparison.

Score: 0.0 (wrong) to 1.0 (perfect). 
Output ONLY the float value."""

    raw_score = raw_score.strip().rstrip('.')

    match = re.search(r"(0?\.\d+|1\.0|0|1)", raw_score)
    
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    print(f"⚠️ Judge parse failed. Raw: '{raw_score}'")
    # fallback
    if contains_expected(answer, expected_answer):
        return 0.8
    sim = compute_similarity(answer, expected_answer)
    return sim if sim is not None else 0.0

def contains_expected(answer: str, expected: str, threshold: float = 0.7) -> bool:
    sim = compute_similarity(expected, answer)
    if sim is None:
        return False
    return sim > threshold

def check_consistency(
    question: str,
    query_fn: Callable[[str], str],
    num_retries: int = 2
) -> Optional[float]:
    if question is None or not str(question).strip():
        return None
    
    if not callable(query_fn):
        return None
    
    try:
        # Get multiple answers
        answers = []
        for _ in range(num_retries + 1):
            answer = query_fn(question)
            if answer is not None:
                answers.append(str(answer).strip())
        
        if len(answers) < 2:
            return None
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sim = compute_similarity(answers[i], answers[j])
                if sim is not None:
                    similarities.append(sim)
        
        if not similarities:
            return None
        
        # Return average similarity
        consistency_score = float(np.mean(similarities))
        return max(0.0, min(1.0, consistency_score))
    
    except Exception:
        return None


def flag_failures(
    quality_score,
    consistency_score,
    quality_threshold: float = 0.5,
    consistency_threshold: float = 0.7
):
    issues = []
    if quality_score is not None and quality_score < quality_threshold:
        issues.append('low_quality')
    if consistency_score is not None and consistency_score < consistency_threshold:
        issues.append('low_consistency')
    return issues

def clear_embedding_cache():
    _embedding_cache.clear()
