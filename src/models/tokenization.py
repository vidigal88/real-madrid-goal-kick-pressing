"""Zone-based tokenization of pressing movements for topic modeling.

This module converts continuous pressing movements into discrete token sequences by
mapping player positions to learned spatial zones (via GMM). Each pressing action is
represented as a zone transition token (e.g., "Zone 3 → Zone 11"), enabling
sequence-based pattern discovery through NMF topic modeling.

**Tokenization Concept:**

Instead of analyzing raw (x, y) trajectories, we represent pressing as:
- **Initial Zone**: Where presser starts (kick + 1s) → 1 of 8 zones
- **Target Zone**: Where presser moves to (kick + 5s) → 1 of 15 zones
- **Token**: Transition pair (init_zone, target_zone) → 1 of 120 possible tokens

Example: Player moves from Zone 3 (left defensive midfield) to Zone 11 (central attacking midfield)
→ Token ID = 3 * 15 + 11 = 56

**Why Tokens?**

- **Dimensionality Reduction**: 120 tokens vs. infinite continuous coordinates
- **Interpretability**: "Zone 3→11" more meaningful than "(-35.2, 5.1) → (-18.7, 2.3)"
- **Pattern Discovery**: NMF can find pressing "motifs" (e.g., coordinated left-side press)
- **Comparison**: Easy to compare build-ups ("both have high weight for Token 56")

**Soft Assignment (Probabilistic Weighting):**

Players near zone boundaries get fractional token weights based on GMM probabilities:
- Player 70% in Zone 3, 30% in Zone 2 initially
- Player 60% in Zone 11, 40% in Zone 10 at target
- Contributes weights to multiple tokens:
  - Token(3→11): 0.70 * 0.60 = 0.42
  - Token(3→10): 0.70 * 0.40 = 0.28
  - Token(2→11): 0.30 * 0.60 = 0.18
  - Token(2→10): 0.30 * 0.40 = 0.12

This captures movement uncertainty and overlapping zone membership.

**Pipeline:**

1. **Load Trained GMMs** (from gmm_zones.py output):
   - gmm_initial.pkl (8 zones)
   - gmm_target.pkl (15 zones)

2. **For Each Build-Up**:
   - Extract presser positions at kick+1s and kick+5s
   - Compute zone probabilities using GMM.predict_proba()
   - Generate token weights via outer product of probabilities
   - Accumulate weights across all pressers

3. **Build Term Matrix**:
   - Rows: Build-ups
   - Columns: Tokens (token_0 to token_119)
   - Values: Aggregated token weights

**Output:**

Term matrix (build_ups × tokens) saved as term_matrix.parquet:
```
         token_0  token_1  token_2  ...  token_119
build_up_id
123         0.42     0.18     0.00  ...       0.85
456         0.00     1.23     0.67  ...       0.00
```

**Usage:**

    # Generate token matrix from extracted build-ups
    $ python src/models/tokenization.py \\
        --processed-root data/processed/rm_pressing \\
        --topics-dir data/processed/rm_pressing_zones \\  # Where GMMs are saved
        --out-dir data/processed/rm_pressing_tokens

    # Output: term_matrix.parquet (N_build_ups × 120 tokens)

**Downstream:**

- **NMF Topic Modeling** (nmf_topics.py): Discovers pressing pattern topics
- **Clustering** (clustering.py): Groups similar pressing sequences
- **Visualization**: Plot token frequency distributions

**See Also:**

- src/models/gmm_zones.py: Zone model training
- src/models/nmf_topics.py: Topic modeling on token matrix
- docs/concepts/tokenization.md: Deep dive on tokenization methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.models.gmm_zones import GMMZoneModel # Assuming wrapper is importable

def tokenize_build_up(initial_pos: np.ndarray, target_pos: np.ndarray, gmm_initial, gmm_target) -> Dict[str, float]:
    """Tokenize a single build-up by mapping presser movements to zone transition tokens.

    Converts N presser movements (initial → target positions) into a weighted token
    distribution. Each token represents a zone-to-zone transition (init_zone → target_zone),
    with weights computed via probabilistic GMM zone assignment.

    **Token ID Encoding:**

    Token ID = init_zone_index * n_target_zones + target_zone_index

    With 8 initial zones and 15 target zones:
    - Token 0: Zone 0 → Zone 0
    - Token 15: Zone 1 → Zone 0
    - Token 16: Zone 1 → Zone 1
    - Token 119: Zone 7 → Zone 14

    **Weight Calculation:**

    For each presser k:
    1. Get initial zone probabilities: p_init[k] = [0.7, 0.3, 0.0, ..., 0.0]  (8 values)
    2. Get target zone probabilities: p_target[k] = [0.0, 0.6, 0.4, ..., 0.0]  (15 values)
    3. Compute token weights via outer product: w[i,j] = p_init[k][i] * p_target[k][j]
    4. Accumulate across all pressers

    Args:
        initial_pos: Array of shape (N_pressers, 2) with (x, y) coordinates at kick+1s.
            Typically normalized coordinates (opponent attacks right).
        target_pos: Array of shape (N_pressers, 2) with (x, y) coordinates at kick+5s.
            Must have same N_pressers as initial_pos (matched pairs).
        gmm_initial: Trained sklearn GaussianMixture for initial zones (8 components).
        gmm_target: Trained sklearn GaussianMixture for target zones (15 components).

    Returns:
        Dictionary mapping token IDs to aggregated weights:
        {
            'token_0': 0.42,
            'token_15': 1.23,
            'token_56': 0.85,
            ...
        }

        Returns empty dict {} if no pressers or prediction fails.

    Example:
        >>> # Single build-up with 3 pressers
        >>> initial_pos = np.array([[-40, 5], [-35, -3], [-38, 0]])
        >>> target_pos = np.array([[-25, 8], [-20, -5], [-22, 2]])
        >>>
        >>> # Load trained GMMs
        >>> import pickle
        >>> with open("gmm_initial.pkl", "rb") as f:
        ...     gmm_init = pickle.load(f)
        >>> with open("gmm_target.pkl", "rb") as f:
        ...     gmm_targ = pickle.load(f)
        >>>
        >>> # Tokenize
        >>> tokens = tokenize_build_up(initial_pos, target_pos, gmm_init, gmm_targ)
        >>> print(f"Build-up represented by {len(tokens)} non-zero tokens")
        >>> print(f"Highest-weight token: {max(tokens, key=tokens.get)}")

    Notes:
        - Soft assignment (probabilistic) ensures smooth transitions near zone boundaries
        - Token weights are unnormalized (sum across tokens may exceed N_pressers)
        - Empty positions arrays return empty dict (graceful degradation)
    """
    token_weights = defaultdict(float)
    
    if len(initial_pos) == 0 or len(target_pos) == 0:
        return {}
        
    # Get probabilities
    try:
        # predict_proba returns (N_samples, N_components)
        p_init = gmm_initial.predict_proba(initial_pos)
        p_target = gmm_target.predict_proba(target_pos)
    except Exception:
        # Fallback if prediction fails
        return {}
        
    n_init = p_init.shape[1]
    n_target = p_target.shape[1]
    
    # For each presser
    for k in range(len(initial_pos)):
        # Presser k
        # We form tokens from Init Zone i -> Target Zone j
        # Weight = P(init=i) * P(target=j)
        
        pi = p_init[k] # shape (n_init,)
        pt = p_target[k] # shape (n_target,)
        
        # Outer product to get all pairs (i, j) weights
        # weight_matrix[i, j] = pi[i] * pt[j]
        w_matrix = np.outer(pi, pt)
        
        # Accumulate to tokens
        for i in range(n_init):
            for j in range(n_target):
                token_id = i * n_target + j
                token_weights[f"token_{token_id}"] += w_matrix[i, j]
                
    return dict(token_weights)

def main() -> None:
    """Main entry point for tokenizing all build-ups from command line.

    Loads GMM models, processes all build-ups, generates token representations,
    and exports to parquet file for NMF topic modeling.
    """
    import argparse
    import pickle
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from src.features.services.window_loader import WindowLoader
    from src.features.services.normalization import normalize_coordinates
    from src.features.services.possession import infer_ball_carrier
    from src.features.services.utils import time_to_seconds
    from src.models.gmm_zones import identify_pressers
    from src.models.config import GMMConfig
    from src.utils.pickle_compat import load_pickle_compat_or_raise
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="data/processed/rm_pressing")
    parser.add_argument("--topics-dir", default="data/processed/rm_pressing_topics")
    parser.add_argument("--out-dir", default="data/processed/rm_pressing_tokens")
    args = parser.parse_args()
    
    topics_dir = Path(args.topics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GMMs
    try:
        gmm_initial = load_pickle_compat_or_raise(topics_dir / "gmm_initial.pkl")
        gmm_target = load_pickle_compat_or_raise(topics_dir / "gmm_target.pkl")
    except FileNotFoundError:
        print("GMM models not found. Run gmm_zones.py first.")
        return

    loader = WindowLoader(args.processed_root)
    ids = loader.index["build_up_id"].tolist()
    gmm_config = GMMConfig()
    
    # Prepare matrix builder
    rows = []
    
    for bid in tqdm(ids, desc="Tokenizing"):
        try:
            # 1. Load & Process (Duplicate logic from GMM prep - should refactor but fine for now)
            df = loader.load_build_up(bid)
            meta = loader.get_metadata(bid)
            
            # Preprocess
            from src.features.services.utils import prepare_frame_data
            df = prepare_frame_data(df)
            
            # Enrich
            from src.features.services.metadata import enrich_with_team_id
            df = enrich_with_team_id(df, meta["game_id"])
            
            df_norm = normalize_coordinates(df, meta.get("gk_side", "left"))
            df_norm = infer_ball_carrier(df_norm, meta.get("opponent_team_id"))
            
            if "time_seconds" not in df_norm.columns:
                df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
                
            pressers = identify_pressers(df_norm, gmm_config)
            
            # Extract positions
            t_kick = time_to_seconds(str(meta.get("kick_time")))
            t_init = t_kick + 1.0
            t_target = t_kick + 5.0
            
            # Simplify position extraction (Copy-paste logic :()
            # In a real app, `extract_positions` should be a shared service function.
            # I will trust the agent to refactor or just inline.
            
            def get_pos_batch(t, pids, df):
                times = df[["frame", "time_seconds"]].drop_duplicates()
                ft_idx = (times["time_seconds"] - t).abs().idxmin()
                best_frame = times.loc[ft_idx, "frame"]
                frame_df = df[df["frame"] == best_frame]
                
                pos_list = []
                for pid in pids:
                    p_row = frame_df[frame_df["player_id"] == pid]
                    if not p_row.empty:
                        pos_list.append([p_row.iloc[0]["x_norm"], p_row.iloc[0]["y_norm"]])
                return np.array(pos_list)

            p_init = get_pos_batch(t_init, pressers, df_norm)
            p_target = get_pos_batch(t_target, pressers, df_norm)
            
            # Tokenize
            tokens = tokenize_build_up(p_init, p_target, gmm_initial, gmm_target)
            
            # Add to list
            tokens["build_up_id"] = bid
            rows.append(tokens)
            
        except Exception as e:
            continue
            
    # Create Matrix
    term_df = pd.DataFrame(rows)
    term_df.set_index("build_up_id", inplace=True)
    term_df.fillna(0.0, inplace=True)
    
    # Save
    term_df.to_parquet(out_dir / "term_matrix.parquet")
    print(f"Saved term matrix with shape {term_df.shape}")

if __name__ == "__main__":
    main()

