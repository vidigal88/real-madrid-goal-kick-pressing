"""Narrative generation for NMF topic interpretation.

This module transforms NMF topic models (W and H matrices) into human-readable
textual narratives. Narratives facilitate tactical interpretation by:
- Describing top pressing patterns (zone transitions) in each topic
- Identifying representative build-ups exhibiting each topic
- Linking topics to pressing archetypes (high press, containment, etc.)

**Narrative Structure:**

For each topic:
1. **Header**: Topic ID and separator
2. **Top Pressing Patterns**: 5 highest-weight tokens (zone transitions)
   - Decoded as "Zone X → Zone Y (weight=0.82)"
   - Sorted by weight (descending)
3. **Representative Build-ups**: 3 build-ups with highest topic loading
   - Build-up IDs and H-matrix values
4. **Interpretation** (optional): Tactical description of topic archetype

**Usage:**

    from src.analysis.narratives import generate_topic_narratives

    W = pd.read_parquet("W.parquet")  # Token-topic weights
    H = pd.read_parquet("H.parquet")  # Build-up-topic weights
    token_dict = create_token_dictionary()  # Token → (init_zone, target_zone)
    features_df = pd.read_parquet("features.parquet")

    report = generate_topic_narratives(W, H, token_dict, features_df)
    print(report)

    # Output:
    # Topic 0
    # ====================
    # Top Pressing Patterns:
    #   - Zone 3 -> Zone 11 (w=0.82)
    #   - Zone 2 -> Zone 12 (w=0.71)
    #   ...
    # Representative Build-ups:
    #   - #123 (load=0.85)
    #   - #456 (load=0.78)
    #   ...

**Applications:**

- **Tactical Reports**: Automatic topic interpretation for analysts
- **Pattern Discovery**: Quickly understand pressing archetypes
- **Build-Up Analysis**: Find examples of specific pressing strategies
- **Coaching Communication**: Translate model output to coach-friendly language

**See Also:**

- src/models/nmf_topics.py: Generates W and H matrices
- src/models/tokenization.py: Creates token representations
- docs/concepts/nmf-topics.md: Topic modeling methodology
"""

import pandas as pd
from typing import Dict, List, Any

def generate_topic_narratives(W: pd.DataFrame, H: pd.DataFrame, token_dict: pd.DataFrame, features_df: pd.DataFrame) -> str:
    """Generate textual narrative describing NMF topics and their characteristics.

    Creates a structured report for each topic, listing top pressing patterns
    (zone transitions) and representative build-ups. Facilitates qualitative
    interpretation of quantitative topic models.

    Args:
        W: Token-topic weights matrix, shape (120 tokens, K topics).
           Index: token IDs (token_0 to token_119)
           Columns: Topic IDs
           Values: Weights indicating token importance in each topic

        H: Build-up-topic weights matrix, shape (K topics, N build_ups).
           Index: Topic IDs
           Columns: build_up_id
           Values: Topic loadings (how much each topic appears in build-up)

        token_dict: DataFrame mapping token IDs to zone transitions.
           Index: token IDs
           Columns: 'init_zone', 'target_zone'
           Used to decode token IDs into interpretable transitions

        features_df: Build-up features DataFrame (currently unused, reserved
           for future enhancements like adding opponent/outcome context)

    Returns:
        Multi-line string containing formatted narrative for all topics.
        Each topic section includes:
        - Topic ID header
        - Top 5 tokens with weights
        - Top 3 representative build-ups with loadings

    Example:
        >>> W = pd.read_parquet("W.parquet")
        >>> H = pd.read_parquet("H.parquet")
        >>> token_dict = pd.DataFrame({
        ...     'init_zone': [3, 2, 5],
        ...     'target_zone': [11, 12, 3]
        ... }, index=['token_56', 'token_42', 'token_78'])
        >>>
        >>> narrative = generate_topic_narratives(W, H, token_dict, None)
        >>> print(narrative)
        Topic 0
        ====================
        Top Pressing Patterns:
          - 3 -> 11 (w=0.82)
          - 2 -> 12 (w=0.71)
          ...

    Notes:
        - Topics are unordered; manual labeling needed post-generation
        - Token weights indicate importance but not frequency
        - Representative build-ups show high topic purity (dominated by one topic)
    """
    report = []
    
    # Iterate topics
    for topic_id in W.columns:
        report.append(f"Topic {topic_id}")
        report.append("="*20)
        
        # Top tokens
        top_tokens = W[topic_id].nlargest(5)
        report.append("Top Pressing Patterns:")
        for tid, weight in top_tokens.items():
            # Resolve tid to zones
            # token_dict usually has 'init_zone', 'target_zone'.
            # If token_dict is provided.
            if tid in token_dict.index:
                row = token_dict.loc[tid]
                report.append(f"  - {row['init_zone']} -> {row['target_zone']} (w={weight:.2f})")
            else:
                report.append(f"  - Token {tid} (w={weight:.2f})")
                
        # Representative Build-ups (High H)
        top_bur = H.loc[topic_id].nlargest(3)
        report.append("\nRepresentative Build-ups:")
        for bid, val in top_bur.items():
            report.append(f"  - #{bid} (load={val:.2f})")
            
        report.append("\n" + "-"*30 + "\n")
        
    return "\n".join(report)
