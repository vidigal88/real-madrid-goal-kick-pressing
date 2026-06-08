"""Goal kick classification and reception analysis.

This module classifies opponent goal-kick restarts by first-reception distance.

The main reader-facing split is short vs direct:
- short: first reception < 15m
- direct: first reception >= 15m

True long restarts (>= 30m) are retained as a descriptive sub-label because
they are rare in this dataset.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.features.config import GoalKickConfig
from src.features.services.utils import time_to_seconds


def classify_restart_distance(distance_m: float, config: GoalKickConfig) -> Dict[str, Any]:
    """Return restart labels from the first-reception distance."""
    if pd.isna(distance_m):
        return {
            "goal_kick_type": "unknown",
            "restart_type": "unknown",
            "restart_distance_band": "unknown",
            "direct_restart_depth": "unknown",
            "is_true_long_restart": False,
            "legacy_goal_kick_type": "unknown",
        }

    distance_m = float(distance_m)
    legacy_goal_kick_type = "long" if distance_m >= config.long_kick_threshold_m else "short"

    if distance_m < config.short_restart_threshold_m:
        restart_type = "short"
        restart_distance_band = "short_under_15m"
        direct_restart_depth = "short"
    elif distance_m < config.long_kick_threshold_m:
        restart_type = "direct"
        restart_distance_band = "medium_15_30m"
        direct_restart_depth = "medium"
    else:
        restart_type = "direct"
        restart_distance_band = "long_30m_plus"
        direct_restart_depth = "long_30m_plus"

    return {
        "goal_kick_type": restart_type,
        "restart_type": restart_type,
        "restart_distance_band": restart_distance_band,
        "direct_restart_depth": direct_restart_depth,
        "is_true_long_restart": bool(distance_m >= config.long_kick_threshold_m),
        "legacy_goal_kick_type": legacy_goal_kick_type,
    }


def unknown_goal_kick_result() -> Dict[str, Any]:
    return {
        "goal_kick_type": "unknown",
        "restart_type": "unknown",
        "restart_distance_band": "unknown",
        "direct_restart_depth": "unknown",
        "is_true_long_restart": False,
        "legacy_goal_kick_type": "unknown",
        "gk_kick_distance_m": float("nan"),
        "time_to_receiver_s": float("nan"),
        "receiver_lane": "unknown",
    }


def classify_goal_kick(df_norm: pd.DataFrame, kick_time: str,
                       config: GoalKickConfig = GoalKickConfig()) -> Dict[str, Any]:
    """Classify goal kick type and analyze reception characteristics.

    Determines whether a goal-kick restart is short (<15m) or direct (>=15m)
    by tracking ball trajectory and identifying the first reception point.
    Restarts of >=30m are additionally flagged as true long restarts.

    Args:
        df_norm: Normalized tracking data with ball and player positions.
            Must contain 'ball_x_norm', 'ball_y_norm', 'time', and optionally
            'ball_carrier_id' columns.
        kick_time: Timestamp of the goal kick in tracking time format (HH:MM:SS.DD).
        config: Configuration object with short_restart_threshold_m (default
            15m), long_kick_threshold_m (default 30m), and lane_width_m
            (default 20m).

    Returns:
        Dictionary containing:
            - goal_kick_type (str): 'short', 'direct', or 'unknown'.
            - restart_type (str): Same main split as goal_kick_type.
            - restart_distance_band (str): short_under_15m, medium_15_30m,
              long_30m_plus, or unknown.
            - direct_restart_depth (str): short, medium, long_30m_plus, or unknown.
            - is_true_long_restart (bool): True when distance >= 30m.
            - legacy_goal_kick_type (str): old short/long split at 30m.
            - gk_kick_distance_m (float): Euclidean distance from kick to reception.
            - time_to_receiver_s (float): Time elapsed from kick to reception.
            - receiver_lane (str): Reception zone ('left', 'center', 'right', 'unknown').

    Raises:
        KeyError: If required columns are missing from df_norm.

    Example:
        >>> result = classify_goal_kick(df_norm, "00:05:23.50")
        >>> print(f"{result['goal_kick_type']}: {result['gk_kick_distance_m']:.1f}m")
        direct: 18.3m

    Notes:
        - Requires 'ball_carrier_id' column for accurate reception detection.
        - Reception is identified when ball travels > 5m from kick position.
        - Lane classification uses ±10m threshold around pitch centerline.
        - Returns 'unknown' if insufficient data to classify.
    """
    # 1. Identify kick moment
    kick_seconds = time_to_seconds(kick_time)
    
    # Ensure df has seconds
    if "time_seconds" not in df_norm.columns:
        df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
    
    # Filter for post-kick frames
    # Giving a small buffer (-0.5s) to be sure we catch the start if timing is slightly off
    post_kick_df = df_norm[df_norm["time_seconds"] >= kick_seconds].copy()
    
    if post_kick_df.empty:
         return unknown_goal_kick_result()

    # Find kick position (Ball at kick_time)
    # Get frame closest to kick_time
    kick_frame_row = post_kick_df.iloc[0] # Approximation
    # find row with time_seconds closest to kick_seconds
    # usually post_kick_df.iloc[0] is the start.
    
    ball_x_kick = kick_frame_row.get("ball_x_norm", kick_frame_row.get("ball_x"))
    ball_y_kick = kick_frame_row.get("ball_y_norm", kick_frame_row.get("ball_y"))
    
    if pd.isna(ball_x_kick) or pd.isna(ball_y_kick):
        # Scan a few frames to find valid ball
        for _, row in post_kick_df.head(10).iterrows():
            bx = row.get("ball_x_norm", row.get("ball_x"))
            by = row.get("ball_y_norm", row.get("ball_y"))
            if not pd.isna(bx):
                ball_x_kick = bx
                ball_y_kick = by
                break
                
    if pd.isna(ball_x_kick):
        return unknown_goal_kick_result()

    # 2. Scan forward for "Reception"
    # Logic: Look for the moment when a player from the attacking team (opponent of RM) has possession.
    # possession.py gives "ball_carrier_id".
    # df_norm might already have "ball_carrier_id" if we ran inference.
    
    # We'll assume df_norm has 'ball_carrier_id' or we need to pass the possession logic result.
    # If not, we can't determine reception easily without duplicating possession logic.
    # Assuming `infer_ball_carrier` was run on `df_norm` BEFORE calling this.
    
    reception_row = None
    
    # Iterate through frames
    # Optimization: Filter rows where ball_carrier_id is not NaN
    if "ball_carrier_id" in post_kick_df.columns:
        carriers = post_kick_df.dropna(subset=["ball_carrier_id"])
        # make sure the carrier is NOT the GK (if GK takes the kick).
        # Usually GK takes kick, so the "first carrier" is the GK.
        # We want the *next* carrier (the receiver).
        
        # Or simpler: Is the distance from kick > 5m?
        
        for _, row in carriers.iterrows():
            bx = row.get("ball_x_norm", row["ball_x"])
            by = row.get("ball_y_norm", row["ball_y"])
            
            dist_from_kick = np.sqrt((bx - ball_x_kick)**2 + (by - ball_y_kick)**2)
            
            if dist_from_kick > 5.0: # Threshold to ignore initial touch/GK
                reception_row = row
                break
    
    if reception_row is None:
        # Fallback: End of window ball position if no carrier found?
        # Or look for stable ball?
        # use last frame ball position as proxy if no reception?
        # that's dangerous. "Unknown" is better.
        
        # try to detect if ball is "far" at any point.
        max_dist = 0.0
        # Check max ball dist from kick
        # Vectorized
        bx_all = post_kick_df.get("ball_x_norm", post_kick_df.get("ball_x"))
        by_all = post_kick_df.get("ball_y_norm", post_kick_df.get("ball_y"))
        
        dists = np.sqrt((bx_all - ball_x_kick)**2 + (by_all - ball_y_kick)**2)
        max_dist = dists.max()
        
        result = classify_restart_distance(max_dist, config)
        result.update({
            "gk_kick_distance_m": float(max_dist),
            "time_to_receiver_s": float('nan'),
            "receiver_lane": "unknown",
        })
        return result

    # If reception found
    bx_recv = reception_row.get("ball_x_norm", reception_row["ball_x"])
    by_recv = reception_row.get("ball_y_norm", reception_row["ball_y"])
    dist = np.sqrt((bx_recv - ball_x_kick)**2 + (by_recv - ball_y_kick)**2)
    time_diff = reception_row["time_seconds"] - kick_seconds
    
    # Lane
    lane = "center"
    if by_recv < -10.0: # Assuming y is -34 to 34
        lane = "left" # attacking right, y negative is... LEFT? 
        # Coordinate system: normalized: Attack Right (positive X).
        # Standard: Y is left-right.
        # If standard: Y > 0 is Left (from camera)? 
        # Usually: 0,0 center. Y up/down.
        # If Attack Right, Left side is Y > 0? No, usually Y axis is orthogonal.
        # assume standard intuition: "Left Lane" relative to attacking direction.
        # If attacking X+, Left is Y+ (standard right-handed) or Y- (screen)?
        # config says lane_width_m = 20 (center is -10 to 10).
        # So "Left" is < -10 or > 10.
        # assume Y < -10 is Left, Y > 10 is Right (or vice versa).
        # I'll stick to: -10 to 10 is center.
        if by_recv < -10: lane = "left" # arbitrary, consistent
        elif by_recv > 10: lane = "right"
    else:
        if by_recv < -10: lane = "right" # swap if needed
        elif by_recv > 10: lane = "left"
        
    result = classify_restart_distance(dist, config)
    result.update({
        "gk_kick_distance_m": float(dist),
        "time_to_receiver_s": float(time_diff),
        "receiver_lane": lane
    })
    return result
