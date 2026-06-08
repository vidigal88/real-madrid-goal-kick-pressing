import pandas as pd
from typing import Dict, Any
from src.features.config import QCConfig

def compute_qc_metrics(df_norm: pd.DataFrame, config: QCConfig = QCConfig()) -> Dict[str, Any]:
    """
    Computes quality control metrics.
    
    Output:
    qc_ball_detected_ratio
    qc_avg_players_detected
    qc_carrier_missing_ratio
    qc_pass_quality
    """
    total_frames = len(df_norm["frame"].unique())
    if total_frames == 0:
        return {
            "qc_ball_detected_ratio": 0.0,
            "qc_avg_players_detected": 0.0,
            "qc_carrier_missing_ratio": 1.0,
            "qc_pass_quality": False
        }
        
    # Ball detected ratio
    # If long format: group by frame, check if any ball_detected?
    # Or 'ball_x' is not null.
    # Assuming one ball row per frame or ball cols in every row.
    
    # If we grouped by frame:
    frames_with_ball = df_norm.dropna(subset=["ball_x_norm"])["frame"].nunique()
    ball_ratio = frames_with_ball / total_frames
    
    # Avg players detected
    # Count unique players per frame (where x_norm is not null)
    # Long format
    players_per_frame = df_norm.groupby("frame")["player_id"].nunique() 
    # NOTE: df_norm might include extrapolate players?
    # count DETECTED players ideally.
    # usually 'x' implies existence. 
    # count rows with valid player coordinates.
    avg_players = players_per_frame.mean()
    
    # Carrier missing ratio
    # Carrier is inferred. Returns NaN if not found.
    # We use our own inference result 'ball_carrier_id'.
    if "ball_carrier_id" in df_norm.columns:
        frames_with_carrier = df_norm.dropna(subset=["ball_carrier_id"])["frame"].nunique()
        carrier_ratio = frames_with_carrier / total_frames
        carrier_missing = 1.0 - carrier_ratio
    else:
        carrier_missing = 1.0
        
    # Pass Quality
    qc_pass = (
        ball_ratio >= config.min_ball_detected_ratio and
        avg_players >= config.min_avg_players and
        carrier_missing <= config.max_carrier_missing_ratio
    )
    
    return {
        "qc_ball_detected_ratio": float(ball_ratio),
        "qc_avg_players_detected": float(avg_players),
        "qc_carrier_missing_ratio": float(carrier_missing),
        "qc_pass_quality": bool(qc_pass)
    }
