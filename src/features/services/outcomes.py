import pandas as pd
import numpy as np
from typing import Dict, Any
from src.features.config import OutcomeConfig
from src.features.services.utils import time_to_seconds

def compute_outcome_proxies(df_norm: pd.DataFrame, rm_team_id: int, config: OutcomeConfig = OutcomeConfig()) -> Dict[str, Any]:
    """
    Computes outcome proxies.
    
    Output:
    ended_by_long_ball
    ended_by_midline_cross
    ended_by_out_of_play_proxy
    ended_by_turnover_proxy
    opponent_progress_to_end_m
    """
    if df_norm.empty:
         return {
            "ended_by_long_ball": False,
            "ended_by_midline_cross": False,
            "ended_by_out_of_play_proxy": False,
            "ended_by_turnover_proxy": False,
            "opponent_progress_to_end_m": 0.0
        }
        
    last_frame = df_norm.iloc[-1]
    
    # 1. Long ball (Displacement > 30m in last 2s)
    # Get last 2s
    if "time_seconds" not in df_norm.columns:
        df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
        
    end_time = df_norm["time_seconds"].iloc[-1]
    last_2s = df_norm[df_norm["time_seconds"] >= end_time - config.outcome_window_s]
    
    if last_2s.empty:
        ended_by_long_ball = False
    else:
        # Check start and end of this window
        start_2s = last_2s.iloc[0]
        end_2s = last_2s.iloc[-1]
        
        bx_s, by_s = start_2s.get("ball_x_norm", start_2s.get("ball_x")), start_2s.get("ball_y_norm", start_2s.get("ball_y"))
        bx_e, by_e = end_2s.get("ball_x_norm", end_2s.get("ball_x")), end_2s.get("ball_y_norm", end_2s.get("ball_y"))
        
        if pd.isna(bx_s) or pd.isna(bx_e):
            ended_by_long_ball = False
        else:
            dist = np.sqrt((bx_e - bx_s)**2 + (by_e - by_s)**2)
            # Long ball > 30m displacement (e.g. huge kick)
            # Or x-displacement specifically? Plan: "Ball x-displacement > 30m"
            # use x-displacement per plan description if specific.
            # Plan says: "ended_by_long_ball: bool, # Ball x-displacement > 30m in last 2s"
            
            x_disp = bx_e - bx_s
            ended_by_long_ball = x_disp > config.long_ball_threshold_m
            
    # 2. Midline cross (x > 0)
    # Opponent attacks Right (positive X).
    # Normalized coords: X from -50 to 50?
    # Usually pitch is -52.5 to 52.5. Midline is 0.
    # If they started back deep and ended past 0...
    # Look at last 2s or just END state?
    # Plan: "Ball crosses x=0 in last 2s"
    # Check if ANY frame in last 2s has ball_x > 0 ?
    # Or just if it ends there?
    # "Ball crosses x=0" usually implies transition.
    # check max(ball_x) in last 2s > 0.
    
    max_x_last_2s = last_2s["ball_x_norm"].max() if not last_2s.empty else -999.0
    ended_by_midline_cross = max_x_last_2s > 0
    
    # 3. Out of play (proxy: |y| > 32m)
    # Check last frame or recent frames.
    # "Ball near touchline (|y| > 32m)"
    # Usually check valid ball positions.
    # If ball goes to 34 (out).
    
    last_valid_ball = df_norm.dropna(subset=["ball_y_norm"]).iloc[-1] if not df_norm.dropna(subset=["ball_y_norm"]).empty else None
    if last_valid_ball is not None:
        ended_by_out_of_play = abs(last_valid_ball["ball_y_norm"]) > config.touchline_threshold_m
    else:
        ended_by_out_of_play = False # No ball data
        
    # 4. Turnover (Carrier switches to RM)
    # Check if last carrier was RM.
    # Carrier ID -> Team ID.
    # We need mapping frame -> carrier_team_id.
    # In `possession.py`, we only computed `ball_carrier_id`.
    # look up team_id of that player.
    # `df_norm` has `player_id` and `team_id` BUT it's potentially long format.
    # If it is long format, we can find team of carrier.
    
    # Efficient look up:
    # Filter valid carriers in last 2s.
    last_carriers = last_2s.dropna(subset=["ball_carrier_id"])
    if last_carriers.empty:
        ended_by_turnover = False
    else:
        # Check the LAST carrier's team.
        last_carrier_row = last_carriers.iloc[-1]
        cid = last_carrier_row["ball_carrier_id"]
        
        # Find team of this player.
        # Searching df_norm for this player_id to get team_id
        # (Assuming player team doesn't change).
        player_row = df_norm[df_norm["player_id"] == cid].iloc[0]
        team_id = player_row["team_id"]
        
        ended_by_turnover = (team_id == rm_team_id)
        
    # 5. Progress
    # Max x_norm reached in window
    opp_progress = df_norm["ball_x_norm"].max()
    
    return {
        "ended_by_long_ball": bool(ended_by_long_ball),
        "ended_by_midline_cross": bool(ended_by_midline_cross),
        "ended_by_out_of_play_proxy": bool(ended_by_out_of_play),
        "ended_by_turnover_proxy": bool(ended_by_turnover),
        "opponent_progress_to_end_m": float(opp_progress)
    }
