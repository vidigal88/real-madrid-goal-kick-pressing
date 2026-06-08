"""Non-Negative Matrix Factorization (NMF) for pressing pattern topic discovery.

This module applies NMF to the token-document matrix to discover latent pressing pattern
"topics" - recurring combinations of zone transitions that characterize different pressing
strategies employed by Real Madrid during opponent build-ups.

**Topic Modeling Concept:**

Given a term matrix T (build_ups × tokens), NMF factorizes it into two matrices:
- **W (tokens × topics)**: How much each token contributes to each topic
- **H (topics × build_ups)**: How much each topic appears in each build-up

Mathematically: T ≈ W × H

Each "topic" is a weighted combination of tokens (zone transitions) that tend to occur
together, representing a distinct pressing pattern archetype.

**Example Topics (Interpretation):**

- **Topic 1** (Left-side Press): High weights for tokens like "Zone2→Zone8", "Zone3→Zone9"
  (defenders pressing from left side toward central attacking zones)

- **Topic 2** (High Press): High weights for tokens like "Zone5→Zone13", "Zone6→Zone14"
  (midfielders pushing high up the pitch)

- **Topic 3** (Reactive Press): Mixed weights indicating less coordinated, reactive pressure

**Why NMF (vs PCA/LDA)?**

- **Non-negativity**: Topic weights are always ≥ 0 (easier to interpret as "presence")
- **Sparsity**: With regularization (alpha, l1_ratio), topics focus on key tokens
- **Additive**: Build-ups can have multiple active topics (e.g., 0.6*Topic1 + 0.4*Topic2)
- **Interpretability**: Topics directly correspond to token combinations

**Matrix Dimensions:**

Input:
- T: (N_build_ups × 120 tokens)  # From tokenization pipeline

Output:
- W: (120 tokens × K topics)     # Topic-token weights
- H: (K topics × N_build_ups)    # Build-up-topic weights

Example: K=15 topics, N=94 build-ups
- W: (120 × 15) - "Topic 3 has weight 0.82 for token_56"
- H: (15 × 94) - "Build-up 123 has weight 0.65 for Topic 3"

**Sklearn Coordinate Mapping:**

Sklearn NMF uses (samples × features) convention:
- Input X: (N_docs × N_features) = (build_ups × tokens)
- fit_transform returns: (N_docs × K) = Document-topic weights
- components_ attribute: (K × N_features) = Topic-feature weights

To match plan notation (W=tokens×topics, H=topics×build_ups):
- W_plan = sklearn.components_.T  # Transpose topic-token matrix
- H_plan = sklearn.fit_transform(X).T  # Transpose doc-topic matrix

**Usage:**

    # Train NMF topic model on tokenized build-ups
    $ python src/models/nmf_topics.py \\
        --tokens-dir data/processed/rm_pressing_tokens \\
        --out-dir data/processed/rm_pressing_topics \\
        --n-topics 15

    # Output:
    # - W.parquet: Token-topic weights (120 × 15)
    # - H.parquet: Build-up-topic weights (15 × 94)

**Hyperparameters (NMFConfig):**

- n_topics: Number of latent topics (default: 15)
  - Too few: Lose pressing pattern nuance
  - Too many: Overfitting, hard to interpret
  - Rule of thumb: sqrt(N_tokens) ≈ sqrt(120) ≈ 10-20

- alpha: Regularization strength (default: 0.1)
  - Controls sparsity in W and H
  - Higher → sparser topics (fewer tokens per topic)

- l1_ratio: Balance between L1 and L2 regularization (default: 0.0)
  - 0.0 = L2 only (smooth topics)
  - 1.0 = L1 only (very sparse topics)
  - 0.5 = Elastic net

- init: Initialization method ('nndsvd', 'random')
  - 'nndsvd': Deterministic, faster convergence (default)

**Interpreting Topics:**

Top tokens in each topic reveal pressing strategy:

```python
# Load W matrix
W = pd.read_parquet("W.parquet")  # (120 tokens × 15 topics)

# Topic 3's top tokens
topic_3 = W.iloc[:, 3].sort_values(ascending=False).head(5)
print(topic_3)
# token_56    0.82  # Zone 3 → Zone 11 (left def → central att)
# token_42    0.71  # Zone 2 → Zone 12 (left back → right att)
# token_78    0.65  # Zone 5 → Zone 3 (central → left)
# ...

# Interpretation: "Coordinated left-to-center pressing wave"
```

**See Also:**

- src/models/tokenization.py: Creates input term matrix
- src/models/clustering.py: Groups build-ups by topic loadings
- src/analysis/narratives.py: Generates textual topic interpretations
- docs/concepts/nmf-topics.md: Topic modeling methodology deep dive
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from src.models.config import NMFConfig
from typing import Tuple, Any

def fit_nmf(term_matrix: pd.DataFrame, config: NMFConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Fit NMF topic model on build-up token matrix.

    Decomposes the (build_ups × tokens) matrix into two lower-rank matrices representing
    latent pressing pattern topics. Uses sklearn's NMF with configurable regularization.

    **Matrix Notation:**

    Input: T (N_build_ups × 120 tokens)
    Output: W (120 tokens × K topics), H (K topics × N_build_ups)
    Approximation: T ≈ W × H

    **Algorithm:**

    1. Initialize W and H using 'nndsvd' (non-negative SVD) or random
    2. Iteratively optimize ||T - WH||_F^2 + regularization
    3. Return converged W and H matrices

    Args:
        term_matrix: DataFrame with build_ups as rows (index), tokens as columns.
            Shape: (N_build_ups, 120). Values are token weights from tokenization.
            Typical values: 0.0 to ~5.0 (aggregated presser weights per token).

        config: NMFConfig containing:
            - n_topics: Number of topics K (default: 15)
            - alpha: Regularization strength (default: 0.1)
            - l1_ratio: L1/L2 balance (default: 0.0)
            - init: Initialization method (default: 'nndsvd')
            - random_state: Random seed for reproducibility

    Returns:
        Tuple of (W, H, nmf_model):

        - **W** (DataFrame): Token-topic weights. Shape (120, K).
          Index: token column names from term_matrix.
          Columns: Unlabeled (0 to K-1).
          Values: Non-negative weights indicating token importance in each topic.

        - **H** (DataFrame): Build-up-topic weights. Shape (K, N_build_ups).
          Index: Unlabeled (0 to K-1).
          Columns: build_up_id from term_matrix index.
          Values: Non-negative weights indicating topic presence in each build-up.

        - **nmf_model**: Fitted sklearn NMF object (for reconstruction error, etc.)

    Example:
        >>> import pandas as pd
        >>> from src.models.nmf_topics import fit_nmf
        >>> from src.models.config import NMFConfig
        >>>
        >>> # Load term matrix from tokenization
        >>> term_matrix = pd.read_parquet("term_matrix.parquet")
        >>> print(term_matrix.shape)  # (94 build-ups, 120 tokens)
        >>>
        >>> # Fit NMF with 15 topics
        >>> config = NMFConfig(n_topics=15, alpha=0.1)
        >>> W, H, model = fit_nmf(term_matrix, config)
        >>>
        >>> # Analyze Topic 3
        >>> print("Top tokens in Topic 3:")
        >>> print(W.iloc[:, 3].nlargest(5))
        >>>
        >>> # Find build-ups with high Topic 3 weight
        >>> print("\\nBuild-ups dominated by Topic 3:")
        >>> print(H.iloc[3].nlargest(5))

    Notes:
        - Convergence is not guaranteed; may need to increase max_iter in config
        - Check model.reconstruction_err_ to assess fit quality
        - Topics are unordered; manual inspection needed for interpretation
        - Negative values impossible due to non-negativity constraint
    """
    # term_matrix should be (Terms x Docs) = (120 x 94)
    # sklearn NMF expects (Samples x Features) usually?
    # NMF(n_components=K). fit_transform(X) -> W (Samples x K), H (K x Features).
    # IF we want Topics to be patterns of TOKENS.
    # We treat Build-ups as Samples? Or Tokens as Samples?
    # Standard: Documents (Build-ups) are samples. Terms are features.
    # Input matrix shape: (N_docs, N_terms).
    # Then W = (N_docs, K), H = (K, N_terms).
    # H represents "Topics" (distribution over terms).
    
    # BUT Plan says: T = W x H where T is (120 x 94).
    # W: (120 x K), H: (K x 94).
    # This implies T is (Terms x Docs).
    # So we are decomposing the Term-Document matrix directly.
    # Samples = Terms? No.
    
    # If we stick to sklearn standard:
    # X = (N_docs, N_terms).
    # W = (N_docs, K) [Doc-Topic weights]
    # H = (K, N_terms) [Topic-Term weights]
    
    # Plan Notation:
    # W: topic-token weights (120 x K). (Compatible with H^T in sklearn if X=Docs x Terms)
    # H: build-up-topic weights (K x 94).
    
    # align with sklearn.
    # X = Build-ups x Tokens (94 x 120).
    # nmf = NMF(n_components=K)
    # W_sklearn = nmf.fit_transform(X) -> (94 x K). (This is Build-up x Topics). This matches Plan's H (transposed).
    # H_sklearn = nmf.components_    -> (K x 120). (This is Topic x Tokens). This matches Plan's W (transposed).
    
    # So:
    # Plan W (Tokens x Topics) = H_sklearn.T
    # Plan H (Topics x Build-ups) = W_sklearn.T
    
    # Input `term_matrix` might be (Tokens x Build-ups) or (Build-ups x Tokens).
    # Tokenization.py produces `term_matrix.parquet`. 
    # Usually it's better to store as (Build-ups x Tokens) for pandas.
    
    # assume input is (Build-ups x Tokens).
    X = term_matrix
    
    nmf = NMF(
        n_components=config.n_topics,
        init=config.init,
        alpha_W=config.alpha, # older sklearn uses alpha, newer uses alpha_W/H. 
        # Check sklearn version 1.5.0
        # In 1.5, `alpha` is deprecated or removed? 
        # `alpha_W` and `alpha_H` are used. Or just `alpha` if it maps.
        # use `alpha_W=config.alpha, alpha_H=config.alpha` or check docs.
        # Safe bet: pass kwargs or handle version.
        l1_ratio=config.l1_ratio,
        random_state=config.random_state
    )
    
    # Try-catch for alpha vs alpha_W?
    # Recent sklearn: alpha is deprecated in 1.0, removed in 1.2?
    # "alpha_W" and "alpha_H" are the parameters.
    
    W_sklearn = nmf.fit_transform(X) # (Docs x Topics)
    H_sklearn = nmf.components_      # (Topics x Terms)
    
    # Return matched to Plan's W, H
    # W_plan (Terms x Topics) = H_sklearn.T
    w_plan = pd.DataFrame(H_sklearn.T, index=X.columns)
    # H_plan (Topics x Docs) = W_sklearn.T
    h_plan = pd.DataFrame(W_sklearn.T, columns=X.index)
    
    return w_plan, h_plan, nmf

def main() -> None:
    """Main entry point for NMF topic modeling from command line.

    Loads token matrix, fits NMF model, and exports W and H matrices
    for topic analysis and interpretation.
    """
    import argparse
    from pathlib import Path
    from src.models.config import NMFConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-dir", default="data/processed/rm_pressing_tokens")
    parser.add_argument("--out-dir", default="data/processed/rm_pressing_topics")
    parser.add_argument("--n-topics", type=int, default=15)
    args = parser.parse_args()
    
    tokens_dir = Path(args.tokens_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    matrix_path = tokens_dir / "term_matrix.parquet"
    if not matrix_path.exists():
        print("Term matrix not found.")
        return
        
    term_matrix = pd.read_parquet(matrix_path)
    # Ensure build-ups as rows?
    # Our tokenization produced DataFrame where index=build_up_id, columns=tokens.
    # So it is (Docs x Terms). fit_nmf expects (Docs x Terms).
    
    config = NMFConfig(n_topics=args.n_topics)
    W, H, model = fit_nmf(term_matrix, config)
    
    W.to_parquet(out_dir / "W.parquet")
    H.to_parquet(out_dir / "H.parquet")
    print("Saved NMF topics (W) and loadings (H).")

if __name__ == "__main__":
    main()

