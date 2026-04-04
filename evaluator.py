from sentence_transformers import SentenceTransformer, util
from typing import Optional, Callable, List
import numpy as np


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputation."""
    
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
    """
    Compute cosine similarity between two texts using sentence embeddings.
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        Similarity score between 0 and 1, or None if inputs are invalid
    """
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


def evaluate_answer_quality(answer: str, expected_answer: Optional[str] = None) -> Optional[float]:
    """
    Evaluate the quality of an answer by comparing it to an expected answer.
    
    Args:
        answer: The generated or provided answer
        expected_answer: The reference/expected answer. If None, returns None.
    
    Returns:
        Quality score between 0 and 1, or None if no expected answer provided
    """
    if expected_answer is None:
        return None
    
    if answer is None:
        return 0.0
    
    similarity = compute_similarity(answer, expected_answer)
    
    # If compute_similarity returns None, treat as low quality
    if similarity is None:
        return 0.0
    
    return similarity


def check_consistency(
    question: str,
    query_fn: Callable[[str], str],
    num_retries: int = 2
) -> Optional[float]:
    """
    Check consistency of answers by querying the same question multiple times.
    
    Args:
        question: The question to query
        query_fn: A callable that takes a question and returns an answer
        num_retries: Number of times to query (total queries = num_retries + 1)
    
    Returns:
        Consistency score between 0 and 1 (average pairwise similarity),
        or None if unable to compute
    """
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
    quality_score: Optional[float],
    consistency_score: Optional[float],
    quality_threshold: float = 0.5,
    consistency_threshold: float = 0.7
) -> List[str]:
    """
    Flag failures based on quality and consistency scores.
    
    Args:
        quality_score: Answer quality score (0-1) or None
        consistency_score: Answer consistency score (0-1) or None
        quality_threshold: Minimum acceptable quality score
        consistency_threshold: Minimum acceptable consistency score
    
    Returns:
        List of issue strings detected (e.g., ['low_quality', 'low_consistency'])
    """
    issues = []
    
    if quality_score is not None and quality_score < quality_threshold:
        issues.append('low_quality')
    
    if consistency_score is not None and consistency_score < consistency_threshold:
        issues.append('low_consistency')
    
    return issues


def clear_embedding_cache():
    """Clear the embedding cache to free memory."""
    _embedding_cache.clear()
