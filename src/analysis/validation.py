"""NMF topic model validation metrics.

This module provides quantitative metrics for evaluating NMF topic quality,
including topic coherence (internal consistency) and topic diversity
(distinctiveness across topics).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_topic_coherence(W: pd.DataFrame, token_dict: pd.DataFrame,
                            top_k: int = 10) -> float:
    """Compute coherence score for NMF topics based on token co-occurrence.

    Measures whether top tokens in each topic form semantically coherent
    patterns. Higher coherence indicates tokens frequently appear together
    (e.g., transitions starting from the same zone).

    Args:
        W: Token-topic weights matrix, shape (N tokens, K topics).
           Index: token IDs. Columns: topic IDs.
        token_dict: DataFrame mapping token IDs to zone transitions.
           Must have columns 'init_zone' and 'target_zone'.
        top_k: Number of top tokens per topic to consider for coherence
           calculation. Default 10.

    Returns:
        Coherence score in range [0, 1]. Higher values indicate better
        topic coherence.

    Notes:
        Current implementation returns placeholder value (0.5).
        Full implementation requires co-occurrence matrix computation.

    Example:
        >>> W = pd.read_parquet("W.parquet")
        >>> token_dict = create_token_dictionary()
        >>> coherence = compute_topic_coherence(W, token_dict, top_k=10)
        >>> print(f"Coherence: {coherence:.3f}")
        Coherence: 0.500
    """
    # Placeholder for coherence logic
    # Full implementation would compute co-occurrence patterns
    return 0.5


def compute_topic_diversity(W: pd.DataFrame, top_k: int = 10) -> float:
    """Compute diversity score measuring topic distinctiveness.

    Measures overlap in top-k tokens across topic pairs. Lower overlap
    indicates more diverse topics with distinct pressing patterns.

    Args:
        W: Token-topic weights matrix, shape (N tokens, K topics).
           Index: token IDs. Columns: topic IDs.
        top_k: Number of top tokens per topic to consider for diversity
           calculation. Default 10.

    Returns:
        Diversity score in range [0, 1]. Higher values indicate more
        distinct topics (less overlap in top tokens).
        - 1.0: Perfect diversity (no shared top tokens)
        - 0.0: No diversity (all topics have identical top tokens)

    Algorithm:
        1. For each topic, identify top-k highest-weight tokens
        2. For each pair of topics, compute Jaccard overlap:
           overlap = |intersection| / k
        3. Return: diversity = 1 - mean(overlap across all pairs)

    Example:
        >>> W = pd.read_parquet("W.parquet")
        >>> diversity = compute_topic_diversity(W, top_k=10)
        >>> print(f"Diversity: {diversity:.3f}")
        Diversity: 0.782
    """
    n_topics = W.shape[1]
    overlap_sum = 0.0
    pair_count = 0
    
    # Get top tokens indices per topic
    top_tokens = {}
    for col in W.columns:
        top_tokens[col] = set(W[col].nlargest(top_k).index)
        
    cols = list(W.columns)
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            t1 = top_tokens[cols[i]]
            t2 = top_tokens[cols[j]]
            intersection = len(t1.intersection(t2))
            overlap_sum += intersection / top_k
            pair_count += 1
            
    if pair_count == 0:
        return 1.0
        
    mean_overlap = overlap_sum / pair_count
    return 1.0 - mean_overlap
