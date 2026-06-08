"""Gaussian Mixture Model (GMM) zone assignment for pressing pattern analysis.

This module implements spatial zone modeling using Gaussian Mixture Models to discretize
the continuous pitch space into interpretable pressing zones. The zone-based representation
enables pattern discovery through sequence analysis (tokenization + topic modeling).

**Conceptual Overview:**

Instead of analyzing continuous (x, y) coordinates, we divide the pitch into soft,
overlapping zones learned from data. Each pressing player's movement can be represented
as a transition between zones (e.g., "Zone 3 → Zone 7"), creating interpretable
pressing sequences.

**Two-Stage Zone Modeling:**

1. **Initial Zones (8 zones)**: Capture starting positions of pressers at kick moment + 1s
   - Learned via GMM on all presser initial positions across build-ups
   - Represents "where pressers begin their movement"

2. **Target Zones (15 zones)**: Capture destination positions at kick + 5s
   - More zones needed as players spread out during pressing engagement
   - Represents "where pressers move to engage opponents"

**Why GMM?**

- **Soft Assignment**: Players near zone boundaries get probabilistic membership
- **Data-Driven**: Zone locations emerge from Real Madrid's actual pressing patterns
- **Flexible**: Elliptical zones capture directional pressing tendencies
- **Interpretable**: Gaussian means become reference points for tactical analysis

**Pipeline:**

1. **Identify Active Pressers** (identify_pressers):
   - Select top-K players closest to ball carrier across build-up
   - Filter by minimum frame count (sustained pressure)

2. **Extract Positions** (extract_presser_moves):
   - Initial: kick_time + 1s (reaction phase)
   - Target: kick_time + 5s (engagement phase)

3. **Fit GMM** (GMMZoneModel):
   - Train separate GMMs on aggregated initial/target positions
   - Save models for tokenization pipeline

**Usage Example:**

    # Train zone models on all build-ups
    $ python src/models/gmm_zones.py \\
        --processed-root data/processed/rm_pressing \\
        --out-dir data/processed/rm_pressing_zones

    # Output:
    # - gmm_initial.pkl (8-zone model)
    # - gmm_target.pkl (15-zone model)

**Downstream Applications:**

- **Tokenization** (tokenization.py): Assign zone labels to movements
- **Topic Modeling** (nmf_topics.py): Discover pressing pattern archetypes
- **Visualization**: Plot zone ellipses on pitch diagrams

**Configuration:**

See GMMConfig in models/config.py for hyperparameters:
- n_initial_zones: Number of initial position zones (default: 8)
- n_target_zones: Number of target position zones (default: 15)
- presser_top_k: Top K closest players considered pressers (default: 5)
- presser_min_frames: Minimum frames to qualify as active presser (default: 10)
- covariance_type: GMM covariance structure ("full", "tied", "diag", "spherical")

**See Also:**

- src/models/tokenization.py: Zone-to-token conversion
- src/models/nmf_topics.py: Topic modeling on zone sequences
- docs/concepts/gmm-zones.md: Deep dive on zone modeling methodology
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple, List, Dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from src.models.config import GMMConfig
import logging

logger = logging.getLogger(__name__)

def identify_pressers(df_norm: pd.DataFrame, config: GMMConfig) -> List[int]:
    """Identify active pressing players based on proximity and sustained engagement.

    Determines which Real Madrid players are actively involved in pressing by analyzing
    their proximity to the ball carrier across the build-up window. Players are considered
    "active pressers" if they maintain top-K proximity for a minimum duration.

    **Algorithm:**

    1. For each frame with a detected ball carrier:
       - Calculate Euclidean distance from all Real Madrid players to ball carrier
       - Rank players by distance (closest first)
       - Select top-K players (default K=5)

    2. Tally frame counts per player:
       - Count how many frames each player appears in top-K

    3. Apply duration threshold:
       - Include player if frame count >= min_frames (default: 10 frames ≈ 1 second)

    **Rationale:**

    - **Top-K Selection**: Focuses on players most likely to directly engage opponent
    - **Duration Filter**: Excludes fleeting proximity (e.g., player running past)
    - **Frame-by-Frame**: Accounts for dynamic pressing where closest players change

    Args:
        df_norm: Normalized tracking DataFrame (long format) with columns:
            - 'frame': Frame number
            - 'player_id': Player identifier
            - 'team_id': Team identifier
            - 'x_norm', 'y_norm': Normalized coordinates
            - 'ball_carrier_id': Ball carrier player ID (from infer_ball_carrier)

        config: GMM configuration containing:
            - presser_top_k: Number of closest players to consider (default: 5)
            - presser_min_frames: Minimum frames for active presser status (default: 10)

    Returns:
        List of player IDs identified as active pressers. Empty list if no ball carrier
        detected or insufficient data.

    Example:
        >>> config = GMMConfig(presser_top_k=5, presser_min_frames=10)
        >>> pressers = identify_pressers(df_normalized, config)
        >>> print(f"Identified {len(pressers)} active pressers: {pressers}")
        Identified 6 active pressers: [1045, 1023, 1067, 1089, 1012, 1034]

    Notes:
        - Requires 'ball_carrier_id' column (computed via infer_ball_carrier)
        - Uses team_id to separate Real Madrid players from opponent
        - Carrier's own teammates are excluded from presser candidates
    """
    if "ball_carrier_id" not in df_norm.columns:
        return []
        
    carrier_frames = df_norm.dropna(subset=["ball_carrier_id"])
    if carrier_frames.empty:
        return []
        
    # Count frames where each player is in top-K closest to ball carrier
    presser_counts = {}
    
    for _, frame_df in carrier_frames.groupby("frame"):
        # Carrier
        carrier_row = frame_df[frame_df["player_id"] == frame_df["ball_carrier_id"].iloc[0]]
        if carrier_row.empty: continue
        
        cx, cy = carrier_row["x_norm"].iloc[0], carrier_row["y_norm"].iloc[0]
        
        # Get all players from opposing team (pressers)
        
        carrier_team = carrier_row["team_id"].iloc[0]
        pressers_df = frame_df[frame_df["team_id"] != carrier_team]
        
        if pressers_df.empty: continue
        
        dists = np.sqrt((pressers_df["x_norm"] - cx)**2 + (pressers_df["y_norm"] - cy)**2)
        # Get top-K closest players
        top_k_indices = dists.nsmallest(config.presser_top_k).index
        top_k_pids = pressers_df.loc[top_k_indices, "player_id"]
        
        for pid in top_k_pids:
            presser_counts[pid] = presser_counts.get(pid, 0) + 1
            
    # Filter by min_frames
    active_pressers = [pid for pid, count in presser_counts.items() if count >= config.presser_min_frames]
    return active_pressers

def extract_presser_moves(df_norm: pd.DataFrame, kick_time: str, active_pressers: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts (initial, target) positions for active pressers.
    Initial: kick_time + 1s.
    Target: kick_time + 5s (or window end).
    """
    # ... logic ...
    # Return arrays of shape (N, 2)
    return np.empty((0, 2)), np.empty((0, 2))

class GMMZoneModel:
    """Dual Gaussian Mixture Model for initial and target pressing zone assignment.

    Trains two separate GMMs to model the spatial distribution of pressing players:
    1. Initial GMM: Zones at pressing start (kick + 1s)
    2. Target GMM: Zones at pressing engagement (kick + 5s)

    Each GMM learns K Gaussian components (zones) from aggregated player positions
    across all build-ups. The learned zone centroids and covariances capture Real
    Madrid's typical pressing deployment patterns.

    Attributes:
        config: GMMConfig containing zone counts and model hyperparameters
        gmm_initial: Sklearn GaussianMixture for initial positions (8 zones default)
        gmm_target: Sklearn GaussianMixture for target positions (15 zones default)

    Example:
        >>> from src.models.gmm_zones import GMMZoneModel
        >>> from src.models.config import GMMConfig
        >>>
        >>> # Prepare training data (N_init x 2, N_target x 2 arrays)
        >>> initial_positions = np.array([[...], [...]])  # (x, y) pairs
        >>> target_positions = np.array([[...], [...]])
        >>>
        >>> # Train model
        >>> config = GMMConfig(n_initial_zones=8, n_target_zones=15)
        >>> model = GMMZoneModel(config)
        >>> model.fit(initial_positions, target_positions)
        >>>
        >>> # Save for tokenization pipeline
        >>> model.save(Path("data/processed/rm_pressing_zones"))
        >>>
        >>> # Assign zones to new positions
        >>> zone_labels = model.gmm_initial.predict(new_positions)
        >>> zone_probs = model.gmm_initial.predict_proba(new_positions)

    See Also:
        - src/models/tokenization.py: Uses saved GMMs to assign zone labels
        - src/models/config.py: GMMConfig with hyperparameters
    """

    def __init__(self, config: GMMConfig = GMMConfig()):
        """Initialize dual GMM zone model with configuration.

        Args:
            config: GMMConfig specifying number of zones, covariance type, random seed
        """
        self.config = config
        self.gmm_initial = GaussianMixture(
            n_components=config.n_initial_zones,
            covariance_type=config.covariance_type,
            random_state=config.random_state
        )
        self.gmm_target = GaussianMixture(
            n_components=config.n_target_zones,
            covariance_type=config.covariance_type,
            random_state=config.random_state
        )

    def fit(self, initial_positions: np.ndarray, target_positions: np.ndarray) -> None:
        """Fit both GMMs on aggregated presser positions.

        Trains the initial and target GMMs on collected position data from all
        build-ups. The models learn zone centroids (Gaussian means) and shapes
        (covariance matrices) that best explain the spatial distribution.

        Args:
            initial_positions: Array of shape (N, 2) containing (x, y) coordinates
                of presser positions at kick + 1s across all build-ups.
            target_positions: Array of shape (M, 2) containing (x, y) coordinates
                of presser positions at kick + 5s across all build-ups.

        Note:
            N and M are typically different as not all pressers reach target time.
            Typical sizes: N ≈ 500-2000 points, M ≈ 400-1800 points for 100 build-ups.
        """
        logger.info(f"Fitting Initial GMM on {len(initial_positions)} points")
        self.gmm_initial.fit(initial_positions)

        logger.info(f"Fitting Target GMM on {len(target_positions)} points")
        self.gmm_target.fit(target_positions)

    def save(self, out_dir: Path) -> None:
        """Save trained GMM models to disk as pickle files.

        Serializes both GMMs for later use in tokenization pipeline. Models are
        saved in sklearn-compatible pickle format.

        Args:
            out_dir: Output directory path. Creates if doesn't exist.

        Output Files:
            - gmm_initial.pkl: Initial zone GMM (8 zones)
            - gmm_target.pkl: Target zone GMM (15 zones)

        Example:
            >>> model.save(Path("data/processed/rm_pressing_zones"))
            # Creates:
            # - data/processed/rm_pressing_zones/gmm_initial.pkl
            # - data/processed/rm_pressing_zones/gmm_target.pkl
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "gmm_initial.pkl", "wb") as f:
            pickle.dump(self.gmm_initial, f)
        with open(out_dir / "gmm_target.pkl", "wb") as f:
            pickle.dump(self.gmm_target, f)

    def visualize(self, out_path: str) -> None:
        """Visualize zone ellipses on pitch diagram (TODO: implementation).

        Planned functionality: Plot Gaussian ellipses (2σ) overlaid on pitch outline
        to show learned zone locations and shapes.

        Args:
            out_path: Output file path for visualization (e.g., "zones.png")
        """
        # Todo: Plot ellipses using matplotlib
        # - Draw pitch outline
        # - For each GMM component, draw 2-sigma ellipse
        # - Label zones by index
        # - Save figure to out_path
        pass


def main() -> None:
    """Main entry point for training GMM zone models from command line.

    Loads build-up data, identifies active pressers, extracts position samples,
    fits GMM models for initial and target zones, and saves trained models.
    """
    import argparse
    from src.features.services.window_loader import WindowLoader
    from src.features.services.normalization import normalize_coordinates
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="data/processed/rm_pressing")
    parser.add_argument("--out-dir", default="data/processed/rm_pressing_topics")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    loader = WindowLoader(args.processed_root)
    config = GMMConfig()
    
    all_initial = []
    all_target = []
    
    ids = loader.index["build_up_id"].tolist()
    
    # Needs feature config for normalization
    # Simplification: Use standard normalization
    
    for bid in tqdm(ids, desc="GMM Prep"):
        try:
            df = loader.load_build_up(bid)
            meta = loader.get_metadata(bid)
            
            # Preprocess
            from src.features.services.utils import prepare_frame_data
            df = prepare_frame_data(df)
            
            # Enrich
            from src.features.services.metadata import enrich_with_team_id
            df = enrich_with_team_id(df, meta["game_id"])
            
            gk_side = meta.get("gk_side", "left")
            
            # Normalize
            # use the normalization service.
            # Assuming default config is fine
            df_norm = normalize_coordinates(df, gk_side)
            
            # Infer ball carrier if not present?
            # WindowLoader loads frames. Frames might NOT have ball_carrier_id if we didn't save it back to frames.
            # `features.parquet` has scalar features.
            # Using `feature_engineering.py`, we didn't save the ANNOTATED frames.
            # So we must re-infer carrier here.
            
            # Re-infer carrier
            # Need PossessionConfig
            from src.features.services.possession import infer_ball_carrier
            opp_id = meta.get("opponent_team_id")
            df_norm = infer_ball_carrier(df_norm, opp_id)
            
            kick_time = str(meta.get("kick_time"))
            
            # Identify pressers
            pressers = identify_pressers(df_norm, config)
            # Logic for positions
            # extract_presser_moves(df_norm, kick_time, pressers) 
            # Implement extract_presser_moves logic inside loop or helper
            
            # Logic:
            # Init: kick + 1s
            # Target: kick + 5s
            
            from src.features.services.utils import time_to_seconds
            t_kick = time_to_seconds(kick_time)
            
            if "time_seconds" not in df_norm.columns:
                df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
                
            t_init = t_kick + 1.0
            t_target = t_kick + 5.0
            
            # Find frames closest to t_init and t_target
            # Robust closest frame finder
            def get_pos(t_query, pids):
                # Filter around t_query
                # Get row with min abs diff
                times = df_norm["time_seconds"]
                idx = (times - t_query).abs().idxmin()
                frame_row = df_norm.loc[idx] # warning: if multiple rows per frame (long format)
                # If long format, df_norm has multiple rows for this time.
                # We need the frame NUMBER.
                frame_num = df_norm.loc[idx, "frame"] # scalar or series if duplicates (but index is unique?)
                # df_norm from read_parquet might have RangeIndex.
                # If df_norm is long format, .loc[idx] is ONE row (one player).
                # WRONG.
                
                # Correct way:
                # Find frame number closest to time.
                frame_times = df_norm[["frame", "time_seconds"]].drop_duplicates()
                ft_idx = (frame_times["time_seconds"] - t_query).abs().idxmin()
                best_frame = frame_times.loc[ft_idx, "frame"]
                
                # Get positions for pids in this frame
                frame_df = df_norm[df_norm["frame"] == best_frame]
                
                positions = []
                for pid in pids:
                    p_row = frame_df[frame_df["player_id"] == pid]
                    if not p_row.empty:
                        positions.append([p_row.iloc[0]["x_norm"], p_row.iloc[0]["y_norm"]])
                    else:
                        # Missing player in target frame?
                        pass
                return positions

            pos_init = get_pos(t_init, pressers)
            pos_target = get_pos(t_target, pressers)
            
            # Link them?
            # We treat them as independent points for GMM fitting per plan?
            # Plan: "Collect all presser positions... Fit two separate GMMs"
            # Yes, independent sets of points.
            
            all_initial.extend(pos_init)
            all_target.extend(pos_target)
            
        except Exception as e:
            logger.warning(f"Error in {bid}: {e}")
            continue
            
    # Convert to numpy
    X_init = np.array(all_initial)
    X_target = np.array(all_target)
    
    if len(X_init) < config.n_initial_zones:
        logger.error("Not enough data points for GMM")
        return

    model = GMMZoneModel(config)
    model.fit(X_init, X_target)
    model.save(out_dir)
    logger.info("GMM models saved.")

if __name__ == "__main__":
    main()

