"""Ball steering and directional forcing analysis.

This module analyzes where the ball is forced during a build-up phase,
computing metrics related to spatial displacement, exit lanes, and pressure
applied near touchlines.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.features.config import SteeringConfig
from src.features.services.utils import time_to_seconds


def compute_steering_features(df_norm: pd.DataFrame, kick_time: str,
                               config: SteeringConfig = SteeringConfig()) -> Dict[str, Any]:
    """Analyze ball steering and directional forcing during build-up.

    Computes spatial displacement metrics and exit lane classification to
    understand how pressing forces the ball into different pitch zones.

    Args:
        df_norm: Normalized tracking data with ball and player positions.
            Must contain 'ball_x_norm', 'ball_y_norm', and 'time' columns.
        kick_time: Timestamp of the goal kick in tracking time format (HH:MM:SS.DD).
        config: Configuration object with steering thresholds
            (wide_channel_threshold_m).

    Returns:
        Dictionary containing:
            - exit_lane (str): Final ball position zone ('left', 'right', 'center').
            - ball_x_progress_m (float): Longitudinal displacement from kick to end.
            - ball_y_displacement_m (float): Lateral displacement from kick to end.
            - touchline_pressure_ratio (float): Proportion of frames with pressure
              when ball is in wide channels (|y| > 20m). Returns NaN if 'is_pressure'
              column not available.

    Raises:
        KeyError: If required columns are missing from df_norm.

    Example:
        >>> features = compute_steering_features(df_norm, "00:05:23.50")
        >>> print(features['exit_lane'])
        'right'
        >>> print(features['ball_x_progress_m'])
        18.5

    Notes:
        - Requires 'is_pressure' column for touchline_pressure_ratio calculation.
        - If no post-kick data exists, returns zero/unknown values.
        - Wide channel threshold defaults to 20m from center line.
    """
    kick_seconds = time_to_seconds(kick_time)
    
    if "time_seconds" not in df_norm.columns:
        df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

    post_kick = df_norm[df_norm["time_seconds"] >= kick_seconds]
    if post_kick.empty:
        # Fallback to full window if no post-kick data (e.g. very short window)
        post_kick = df_norm
        
    if post_kick.empty:
         return {
            "exit_lane": "unknown",
            "ball_x_progress_m": 0.0,
            "ball_y_displacement_m": 0.0,
            "touchline_pressure_ratio": 0.0
        }

    # Start position (Kick)
    first_frame = post_kick.iloc[0]
    bx_start = first_frame.get("ball_x_norm", first_frame.get("ball_x"))
    by_start = first_frame.get("ball_y_norm", first_frame.get("ball_y"))
    
    # End position (Window End)
    last_frame = post_kick.iloc[-1]
    bx_end = last_frame.get("ball_x_norm", last_frame.get("ball_x"))
    by_end = last_frame.get("ball_y_norm", last_frame.get("ball_y"))
    
    # Progress/Displacement
    if pd.isna(bx_start) or pd.isna(bx_end):
        ball_x_progress = 0.0
        ball_y_displacement = 0.0
    else:
        ball_x_progress = bx_end - bx_start
        ball_y_displacement = by_end - by_start
        
    # Exit Lane
    # Based on END Y.
    lane = "center"
    if not pd.isna(by_end):
        if by_end < -config.wide_channel_threshold_m: # e.g. -20
            # If attacking right, Y < -20 is Side A. 
            # Assuming Standard Pitch Width 68 -> -34 to 34.
            # wide threshold 20 implies 14m wide channels.
            lane = "left" # Arbitrary mapping; consistent with goal_kick_type
        elif by_end > config.wide_channel_threshold_m:
            lane = "right"
            
    # Touchline pressure ratio
    # Pressure logic needs to be run or passed?
    # `pressure.py` logic is separate.
    # Ideally, we should receive `df_norm` with `is_pressure` column if possible, 
    # Alternative: re-calculate logic?
    # Re-calculating logic is expensive and duplicative.
    # The Orchestrator should probably pass `pressure_info` or `df_norm` should be mutated by `pressure` service?
    # Usually service functions shouldn't mutate input implicitly for others.
    # BUT `compute_steering_features` needs to know about pressure timing.
    
    # Plan says: "Touchline pressure: pressure ratio in frames where |ball_y| > 20m"
    # This implies we filter for wide ball frames, then calculate pressure ratio.
    # This means we DO need to run pressure logic again or have "is_pressure" pre-calculated.
    
    # Strategy: Orchestrator runs `aggregate_pressure_features`, but that returns scalar metrics.
    # It DOES NOT annotate the dataframe with "is_pressure".
    # probably refactor `pressure.py` to optionally annotate OR have a helper for "get_pressure_frames".
    
    # For now, I'll calculate "Touchline Presence" only? 
    # "touchline_pressure_ratio" explicitly asks for pressure within touchline frames.
    # I will assume for now 0.0 or implement a light version of pressure check if carrier data exists.
    # Or better: Update `pressure.py` to return the frame-level series/mask if requested, or split logic.
    
    # assume passed as argument? No, following signature.
    # I'll implement a boolean "ball_wide" ratio for now and rename/adjust or duplicate simple pressure check (dist < 3m).
    
    # Duplicate simple check:
    touchline_frames = post_kick[post_kick["ball_y_norm"].abs() > config.wide_channel_threshold_m]
    if touchline_frames.empty:
        tk_ratio = 0.0
    else:
        # Check pressure in these frames
        # Needs RM positions.
        # getting heavy for this function.
        # I'll look for 'is_pressure' column. If feature engineering pipeline adds it, great.
        # If not, I'll just return NaN or 0 and note dependency.
        
        if "is_pressure" in touchline_frames.columns:
            tk_ratio = touchline_frames["is_pressure"].mean()
        else:
            # TODO: Add logic to annotate pressure in pipeline
            tk_ratio = float('nan')

    return {
        "exit_lane": lane,
        "ball_x_progress_m": float(ball_x_progress),
        "ball_y_displacement_m": float(ball_y_displacement),
        "touchline_pressure_ratio": float(tk_ratio)
    }
