"""Feature engineering pipeline for Real Madrid pressing analysis.

This module orchestrates the complete feature extraction pipeline, transforming raw tracking
data from extracted build-ups into structured numerical features suitable for machine learning
and statistical analysis.

The pipeline executes the following stages:
1. Data loading from extracted build-up windows
2. Coordinate normalization (opponent always attacks right)
3. Ball carrier inference (possession detection)
4. Feature computation across 6 domains:
   - Goal kick restart classification (short/direct, with true-long flag)
   - Pressing pressure metrics (7 metrics)
   - Team compactness metrics (6 metrics)
   - Ball steering/direction (2 metrics)
   - Pass outcome proxies (2 metrics)
   - Data quality indicators (3 metrics)

Total: 20+ features per build-up, capturing pressing patterns, team shape, and build-up characteristics.

Usage:
    Command Line:
        python -m src.features.feature_engineering \\
            --processed-root data/processed/rm_pressing \\
            --out-dir data/processed/rm_pressing_features \\
            --verbose

    Python API:
        from src.features.feature_engineering import extract_features_for_build_up
        from src.features.services.window_loader import WindowLoader
        from src.features.config import FeatureConfig

        loader = WindowLoader(Path("data/processed/rm_pressing"))
        config = FeatureConfig()
        features = extract_features_for_build_up(build_up_id=1, loader=loader, config=config)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm

from src.features.config import FeatureConfig
from src.features.services.window_loader import WindowLoader
from src.features.services.normalization import normalize_coordinates
from src.features.services.possession import infer_ball_carrier
from src.features.services.goal_kick_type import classify_goal_kick
from src.features.services.pressure import aggregate_pressure_features
from src.features.services.compactness import aggregate_compactness_features
from src.features.services.steering import compute_steering_features
from src.features.services.outcomes import compute_outcome_proxies
from src.features.services.qc import compute_qc_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features_for_build_up(
    build_up_id: int,
    loader: WindowLoader,
    config: FeatureConfig
) -> Dict[str, Any]:
    """Extract 20+ numerical features from a single build-up tracking window.

    Orchestrates the complete feature engineering pipeline for one build-up, executing
    data transformations and computing features across multiple tactical domains.

    Args:
        build_up_id: Unique identifier for the build-up (from index.parquet).
        loader: WindowLoader instance with access to extracted build-up data
            (index.parquet, frames/*.parquet, metadata).
        config: FeatureConfig containing all sub-configs:
            - NormalizationConfig: Pitch dimensions
            - PossessionConfig: Ball carrier inference parameters
            - PressureConfig: Pressure detection thresholds
            - CompactnessConfig: Team shape parameters
            - GoalKickConfig: Kick classification thresholds
            - SteeringConfig: Ball direction parameters
            - OutcomeConfig: Pass outcome thresholds
            - QCConfig: Data quality thresholds

    Returns:
        Dictionary containing 20+ features grouped by domain:

        **Identifiers:**
        - build_up_id, period, game_id

        **Goal Kick Features:**
        - goal_kick_type: "short" or "direct"
        - restart_type: same main short/direct split
        - restart_distance_band: short_under_15m, medium_15_30m, or long_30m_plus
        - is_true_long_restart: True when first reception distance is >=30m
        - legacy_goal_kick_type: old short/long split at 30m
        - gk_kick_distance_m: Distance from GK to first receiver
        - receiver_lane: "left", "center", or "right"

        **Pressure Features (7):**
        - t_first_pressure_s: Time to first pressure
        - pressure_frames_ratio: % of frames under pressure
        - pressure_bursts_n: Number of pressing bursts
        - mean_nearest_defender_to_carrier_dist_m
        - min_nearest_defender_to_carrier_dist_m
        - mean_closing_speed_mps
        - max_closing_speed_mps

        **Compactness Features (6):**
        - rm_width_mean_m, rm_width_min_m
        - rm_length_mean_m, rm_length_min_m
        - rm_hull_area_mean_m2
        - rm_line_height_median_x_mean

        **Steering Features (2):**
        - ball_towards_wide_left, ball_towards_wide_right

        **Outcome Features (2):**
        - ball_forward_movement_mean_m
        - ball_backward_movement_mean_m

        **QC Features (3):**
        - ball_detected_ratio
        - mean_players_per_frame
        - qc_pass_quality (bool)

        On error: Returns {'build_up_id': build_up_id, 'error': str(e)}

    Example:
        >>> from pathlib import Path
        >>> from src.features.feature_engineering import extract_features_for_build_up
        >>> from src.features.services.window_loader import WindowLoader
        >>> from src.features.config import FeatureConfig
        >>>
        >>> loader = WindowLoader(Path("data/processed/rm_pressing"))
        >>> config = FeatureConfig()
        >>> features = extract_features_for_build_up(1, loader, config)
        >>>
        >>> print(f"Restart: {features['restart_type']}")
        >>> print(f"Pressure: {features['pressure_frames_ratio']:.2%}")
    """
    try:
        # 1. Load data
        metadata = loader.get_metadata(build_up_id)
        df = loader.load_build_up(build_up_id)
        
        # Preprocess (Long -> Wide Ball)
        from src.features.services.utils import prepare_frame_data
        df = prepare_frame_data(df)
        
        # Enrich with team_id
        from src.features.services.metadata import enrich_with_team_id
        # Assuming we know raw_root or use default
        # Ideally pass raw_root in config or argument. 
        # Using default "data/raw/RealMadrid" for now as per app.py.
        game_id = metadata["game_id"]
        df = enrich_with_team_id(df, game_id, raw_root="data/raw/RealMadrid")
        
        # 2. Normalize coordinates
        # Metadata keys based on app.py / typical extraction: "gk_side", "kick_time", "rm_team_id", etc.
        # Assuming metadata has 'gk_side'.
        gk_side = metadata.get("gk_side", "left") # Default? Check your data.
        df_norm = normalize_coordinates(df, gk_side, config.normalization)
        
        # 3. Infer possession (Ball Carrier)
        opponent_team_id = metadata.get("opponent_team_id")
        # Note: metadata keys might be different. 
        # app.py usage: row['opponent_team_id']
        df_norm = infer_ball_carrier(df_norm, opponent_team_id, config.possession)
        
        # 4. Extract features
        rm_team_id = metadata.get("rm_team_id")
        kick_time = str(metadata.get("kick_time"))
        
        features = {
            "build_up_id": build_up_id,
            "period": metadata.get("period"),
            "game_id": metadata.get("game_id"),
        }
        
        # Services
        features.update(classify_goal_kick(df_norm, kick_time, config.goal_kick))
        features.update(aggregate_pressure_features(df_norm, rm_team_id, config.pressure))
        features.update(aggregate_compactness_features(df_norm, rm_team_id, config.compactness))
        features.update(compute_steering_features(df_norm, kick_time, config.steering))
        features.update(compute_outcome_proxies(df_norm, rm_team_id, config.outcome))
        features.update(compute_qc_metrics(df_norm, config.qc))
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing build_up_id {build_up_id}: {e}")
        return {"build_up_id": build_up_id, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Real Madrid Pressing Feature Extraction")
    parser.add_argument("--processed-root", type=str, default="data/processed/rm_pressing", help="Path to processed data root")
    parser.add_argument("--out-dir", type=str, default="data/processed/rm_pressing_features", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    loader = WindowLoader(processed_root)
    config = FeatureConfig()
    
    logger.info("Loading index...")
    index_df = loader.index
    build_up_ids = index_df["build_up_id"].tolist()
    
    logger.info(f"Found {len(build_up_ids)} build-ups. Starting extraction...")
    
    results = []
    # Loop
    for bid in tqdm(build_up_ids, desc="Extracting features"):
        feat = extract_features_for_build_up(bid, loader, config)
        results.append(feat)
        
    # Combine
    features_df = pd.DataFrame(results)
    
    # Save
    out_path = out_dir / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    
    logger.info(f"Saved {len(features_df)} rows to {out_path}")
    
    # Validation summary
    if "error" in features_df.columns:
        n_errors = features_df["error"].notna().sum()
        if n_errors > 0:
            logger.warning(f"Encountered {n_errors} errors during extraction.")
            
    if "qc_pass_quality" in features_df.columns:
        n_pass = features_df["qc_pass_quality"].sum()
        logger.info(f"QC Pass Rate: {n_pass}/{len(features_df)} ({n_pass/len(features_df):.1%})")

if __name__ == "__main__":
    main()
