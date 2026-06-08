"""Utility functions for feature engineering services.

This module provides time conversion and data preparation utilities used
across feature engineering services.
"""

import pandas as pd
import sys


def time_to_seconds(tracking_time: str) -> float:
    """Convert tracking time format to total seconds as float.

    Args:
        tracking_time: Time string in format 'HH:MM:SS.DD' where DD is
            centiseconds (hundredths of a second).

    Returns:
        Total time in seconds (float). Returns 0.0 if input is invalid.

    Example:
        >>> time_to_seconds("00:05:23.50")
        323.5
        >>> time_to_seconds("01:30:00.00")
        5400.0
    """
    if not isinstance(tracking_time, str):
        return 0.0
    
    parts = tracking_time.split(":")
    if len(parts) < 3:
        return 0.0
        
    hh, mm, ss_dec = parts
    try:
        if "." in ss_dec:
            ss, dec = (ss_dec.split(".", 1) + ["0"])[:2]
        else:
            ss = ss_dec
            dec = "0"
            
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(dec) / 100.0
    except ValueError:
        return 0.0

def seconds_to_time(seconds: float) -> str:
    """Convert seconds (float) to tracking time format string.

    Args:
        seconds: Total time in seconds. Negative values are clamped to 0.0.

    Returns:
        Time string in format 'HH:MM:SS.DD' where DD is centiseconds.

    Example:
        >>> seconds_to_time(323.5)
        '00:05:23.50'
        >>> seconds_to_time(5400.0)
        '01:30:00.00'
    """
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hh = total_cs // (3600 * 100)
    total_cs -= hh * 3600 * 100
    mm = total_cs // (60 * 100)
    total_cs -= mm * 60 * 100
    ss = total_cs // 100
    cs = total_cs - ss * 100
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{cs:02d}"


def prepare_frame_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform long-format tracking data into player-centric format with ball columns.

    Converts tracking data where ball is represented as rows into a format where
    each player row has ball_x and ball_y columns, enabling per-frame spatial
    calculations.

    Args:
        df: Long-format tracking DataFrame with columns:
            - frame: Frame identifier (int)
            - is_ball: Boolean flag indicating ball rows
            - x, y: Position coordinates
            Additional columns are preserved.

    Returns:
        DataFrame with same player rows but added 'ball_x' and 'ball_y' columns.
        Ball rows are removed. Frame column is guaranteed to exist as column
        (not index).

    Raises:
        KeyError: If 'frame' column cannot be found or recovered from index.
        KeyError: If 'is_ball' column is missing.

    Example:
        >>> df_prepared = prepare_frame_data(df_raw)
        >>> print(df_prepared[['frame', 'player_id', 'x', 'y', 'ball_x', 'ball_y']].head())
           frame  player_id      x      y  ball_x  ball_y
        0      1     123456  -20.3   5.2   -30.1    -2.4
        1      1     123457  -18.5   8.1   -30.1    -2.4
    """
    df = df.copy()
    
    # Ensure frame is a column
    if "frame" not in df.columns:
        # Check if frame is the index or one of the levels
        if df.index.name == "frame" or (hasattr(df.index, 'names') and "frame" in df.index.names):
            df.reset_index(inplace=True)
        else:
            # Try resetting anyway, maybe it's unnamed index
            df.reset_index(inplace=True)
            # If 'frame' still missing but 'index' exists, rename?
            if "frame" not in df.columns and "index" in df.columns:
                # Warning: assuming index column is frame
                # Better to just check if 'frame' exists now.
                pass
                
    if "frame" not in df.columns:
        raise KeyError("Column 'frame' not found in dataframe and could not be recovered from index.")

    # Filter ball
    ball_mask = df["is_ball"]
    ball_df = df[ball_mask].copy()
    ball_df = ball_df.rename(columns={"x": "ball_x", "y": "ball_y"})
    
    # Select cols
    ball_df = ball_df[["frame", "ball_x", "ball_y"]]
    
    # Drop duplicates
    ball_df = ball_df.drop_duplicates(subset=["frame"])
    
    # Filter players
    players_df = df[~ball_mask].copy()
    
    # Merge using frame column (safer than join on index if types mismatch)
    merged = pd.merge(players_df, ball_df, on="frame", how="left")
    
    return merged
