"""Pressing partnership affinity metrics for co-pressing analysis.

This module calculates normalized similarity metrics to quantify pressing partnerships
between players. Unlike raw co-occurrence counts, these metrics account for individual
pressing frequencies to reveal genuine coordinated pressing behavior.

**Partnership Metrics:**

1. **Jaccard Similarity**:
   - Formula: J(A,B) = |frames with both A & B| / |frames with A or B|
   - Range: [0, 1]
   - Interpretation: 1.0 = always press together, 0.0 = never overlap
   - Advantage: Intuitive set-based measure, symmetric

2. **Cosine Similarity**:
   - Formula: cos(A,B) = dot(A,B) / (||A|| * ||B||)
   - Range: [0, 1] for binary vectors
   - Interpretation: Angle between pressing participation vectors
   - Advantage: Standard similarity metric, robust to frequency differences

**Why Normalize?**

Raw co-press counts are misleading:
- Player A presses 80/100 frames, Player B presses 10/100 frames
- They co-press 10 frames
- Raw count: 10 co-presses (seems significant)
- Jaccard: 10 / (80+10-10) = 0.125 (low affinity - B presses rarely)
- Reveals that overlap is coincidental, not coordinated

**Usage:**

Compute affinities from pressing participation vectors:

```python
from src.models.pressing_affinity import PressingAffinityCalculator

calc = PressingAffinityCalculator()

# Example: Player A pressed 50 frames, B pressed 40 frames, co-pressed 30 frames
jaccard = calc.jaccard_similarity(30, 50, 40)  # 0.5
cosine = calc.cosine_similarity(30, 50, 40)    # 0.67

print(f"Jaccard affinity: {jaccard:.2f}")
print(f"Cosine affinity: {cosine:.2f}")
```

**Network Analysis Integration:**

These metrics feed into pressing network graphs:
- Nodes: Players
- Edges: Partnerships with affinity > threshold
- Edge weight: Affinity score
- Reveals coordinated pressing units (e.g., CB-CM partnerships)

**See Also:**

- src/analysis/network_centrality.py: Network analysis using affinity metrics
- src/viz/network.py: Pressing network visualization
- docs/concepts/network-analysis.md: Partnership analysis methodology
"""

from typing import Dict, Tuple
import numpy as np


class PressingAffinityCalculator:
    """Calculates normalized pressing partnership metrics."""

    @staticmethod
    def jaccard_similarity(
        co_press_count: int,
        player_a_total: int,
        player_b_total: int
    ) -> float:
        """
        Jaccard similarity: intersection / union.

        Formula: J(A, B) = |frames where both A and B press| / |frames where A OR B press|
                         = co_press_count / (A_total + B_total - co_press_count)

        Args:
            co_press_count: Number of frames where both players pressed together
            player_a_total: Total frames where player A pressed
            player_b_total: Total frames where player B pressed

        Returns:
            Jaccard similarity score between 0.0 and 1.0

        Examples:
            >>> PressingAffinityCalculator.jaccard_similarity(10, 100, 10)
            0.10  # Low affinity: player A presses much more often
            >>> PressingAffinityCalculator.jaccard_similarity(10, 10, 10)
            1.0   # Perfect affinity: always press together
        """
        union = player_a_total + player_b_total - co_press_count
        if union == 0:
            return 0.0
        return co_press_count / union

    @staticmethod
    def cosine_similarity(
        co_press_count: int,
        player_a_total: int,
        player_b_total: int
    ) -> float:
        """
        Cosine similarity: dot product / (norm_a * norm_b).

        Formula: cos(A, B) = co_press_count / sqrt(A_total * B_total)

        Args:
            co_press_count: Number of frames where both players pressed together
            player_a_total: Total frames where player A pressed
            player_b_total: Total frames where player B pressed

        Returns:
            Cosine similarity score between 0.0 and 1.0

        Examples:
            >>> PressingAffinityCalculator.cosine_similarity(10, 100, 10)
            0.316  # Moderate affinity
            >>> PressingAffinityCalculator.cosine_similarity(10, 10, 10)
            1.0    # Perfect affinity
        """
        denom = np.sqrt(player_a_total * player_b_total)
        if denom == 0:
            return 0.0
        return co_press_count / denom

    @staticmethod
    def calculate_affinity_matrix(
        co_press_counts: Dict[Tuple[int, int], int],
        node_sizes: Dict[int, int],
        method: str = "jaccard"
    ) -> Dict[Tuple[int, int], float]:
        """
        Converts raw co-press counts to affinity scores.

        Args:
            co_press_counts: Raw counts {(pid1, pid2): count}
            node_sizes: Individual press counts {pid: total}
            method: "jaccard" or "cosine" similarity metric

        Returns:
            Affinity scores {(pid1, pid2): score} normalized between 0.0 and 1.0

        Raises:
            ValueError: If method is not "jaccard" or "cosine"

        Example:
            >>> counts = {(1, 2): 10, (1, 3): 5}
            >>> sizes = {1: 100, 2: 10, 3: 50}
            >>> affinities = PressingAffinityCalculator.calculate_affinity_matrix(
            ...     counts, sizes, method="jaccard"
            ... )
            >>> affinities[(1, 2)]
            0.10  # Player 1 and 2 have 10% Jaccard affinity
        """
        if method not in ["jaccard", "cosine"]:
            raise ValueError(f"Unknown method: {method}. Must be 'jaccard' or 'cosine'")

        affinities = {}

        for (pid1, pid2), count in co_press_counts.items():
            total_a = node_sizes.get(pid1, 0)
            total_b = node_sizes.get(pid2, 0)

            # Skip if either player has no presses (shouldn't happen but defensive check)
            if total_a == 0 or total_b == 0:
                affinities[(pid1, pid2)] = 0.0
                continue

            if method == "jaccard":
                score = PressingAffinityCalculator.jaccard_similarity(
                    count, total_a, total_b
                )
            else:  # cosine
                score = PressingAffinityCalculator.cosine_similarity(
                    count, total_a, total_b
                )

            affinities[(pid1, pid2)] = score

        return affinities

    @staticmethod
    def get_top_affinity_pairs(
        affinity_scores: Dict[Tuple[int, int], float],
        top_k: int = 10
    ) -> list[Tuple[Tuple[int, int], float]]:
        """
        Returns the top-K player pairs by affinity score.

        Args:
            affinity_scores: Affinity scores from calculate_affinity_matrix()
            top_k: Number of top pairs to return

        Returns:
            List of ((pid1, pid2), score) tuples, sorted by score descending
        """
        sorted_pairs = sorted(
            affinity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pairs[:top_k]
