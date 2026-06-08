
"""
Goal kick restart detection (context + ball kinematics) - single script.

Detects restart frame for goal kicks using:
  1) ball inside goal area (fallback: penalty area) near one goal
  2) pressing team outside penalty area (rule constraint)
  3) kick spike via ball speed/acceleration (kick_candidate)

Also provides diagnostic plots and per-event diagnosis.

"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Match setup
# =========================
game_id = "1021404"

TRACKING_PATH = f"data/RealMadrid/tracking/{game_id}.json"
DYNAMIC_PATH  = Path(f"data/RealMadrid/dynamic/{game_id}.parquet")
META_PATH     = f"data/RealMadrid/meta/{game_id}.json"

HOME_TEAM_ID = 273         # Bilbao (home)
PRESSING_TEAM_ID = 262     # Real Madrid (pressing)


# =========================
# Config
# =========================
FPS = 25
DT = 1 / FPS
MAX_GAP = 25  # frames (1s @ 25Hz)

# kick thresholds (tune)
KICK_SPEED = 8.0
KICK_ACC = 50.0

# context window around dynamic t0 (seconds)
PRE_S = 5.0
POST_S = 90.0
LOOKAHEAD_FRAMES = 25   # find kick_candidate within 1s after context-valid frame

# optional: ball must be low at kick
MAX_Z_AT_KICK = 1.0

# pitch + areas (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 18.32 / 2    # 9.16

PEN_AREA_DEPTH = 16.5
PEN_AREA_HALF_WIDTH = 40.32 / 2     # 20.16

# spatial tolerance (meters)
BUFFER = 1.0


# =========================
# Helpers: time parsing
# =========================
def parse_hhmmss_to_seconds(s: str) -> float:
    """Parse tracking 'HH:MM:SS.ss' -> seconds."""
    if pd.isna(s):
        return np.nan
    h, m, rest = str(s).split(":")
    return float(h) * 3600 + float(m) * 60 + float(rest)

def parse_mmss_to_seconds(s: str) -> float:
    """Parse dynamic 'MM:SS.S' -> seconds."""
    m, rest = str(s).split(":")
    return float(m) * 60 + float(rest)


# =========================
# Helpers: infer which goal is involved (left/right) by ball proximity
# =========================
def infer_defending_side_from_ball_window(w: pd.DataFrame, pitch_length: float = PITCH_LENGTH) -> str:
    """
    Decide whether the restart is happening near the left goal or right goal
    by comparing ball proximity to each goal line within the window.
    """
    if w["x_i"].notna().sum() == 0:
        # no ball positions available
        return "left"

    L = pitch_length
    left_goal_x = -L / 2
    right_goal_x = L / 2

    min_x = float(w["x_i"].min())
    max_x = float(w["x_i"].max())

    dist_left = abs(min_x - left_goal_x)
    dist_right = abs(max_x - right_goal_x)

    return "left" if dist_left < dist_right else "right"


# =========================
# Helpers: geometry
# =========================
def in_goal_area_buf(x, y, defending_side: str, pitch_length: float = PITCH_LENGTH) -> bool:
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return False
    L = pitch_length
    if defending_side == "left":
        return (-L/2 - BUFFER <= x <= (-L/2 + GOAL_AREA_DEPTH + BUFFER)) and (abs(y) <= GOAL_AREA_HALF_WIDTH + BUFFER)
    else:
        return ((L/2 - GOAL_AREA_DEPTH - BUFFER) <= x <= (L/2 + BUFFER)) and (abs(y) <= GOAL_AREA_HALF_WIDTH + BUFFER)

def in_penalty_area_buf(x, y, defending_side: str, pitch_length: float = PITCH_LENGTH) -> bool:
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return False
    L = pitch_length
    if defending_side == "left":
        return (-L/2 - BUFFER <= x <= (-L/2 + PEN_AREA_DEPTH + BUFFER)) and (abs(y) <= PEN_AREA_HALF_WIDTH + BUFFER)
    else:
        return ((L/2 - PEN_AREA_DEPTH - BUFFER) <= x <= (L/2 + BUFFER)) and (abs(y) <= PEN_AREA_HALF_WIDTH + BUFFER)


def count_pressing_team_in_pen_area(frame_dict: dict, pressing_team_id: int, defending_side: str) -> int:
    """Count pressing-team players inside the penalty area of the defending goal."""
    cnt = 0
    for p in frame_dict.get("player_data", []):
        if p.get("team_id") != pressing_team_id:
            continue
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            continue
        if in_penalty_area_buf(float(x), float(y), defending_side=defending_side):
            cnt += 1
    return cnt


# =========================
# Refs: goal kick events from dynamic
# =========================
@dataclass(frozen=True)
class GoalKickRef:
    game_id: str
    period: int
    event_id: str
    time_start: str  # "MM:SS.S"
    team_id: int

def filter_goal_kick_refs(dynamic_df: pd.DataFrame, team_id: int, game_id: str | int) -> list[GoalKickRef]:
    df = dynamic_df[
        (dynamic_df["game_interruption_before"] == "goal_kick_for")
        & (dynamic_df["team_id"] == team_id)
    ][["event_id", "time_start", "period", "team_id"]].copy()

    refs: list[GoalKickRef] = []
    for _, r in df.iterrows():
        refs.append(
            GoalKickRef(
                game_id=str(game_id),
                period=int(r["period"]),
                event_id=str(r["event_id"]),
                time_start=str(r["time_start"]),
                team_id=int(r["team_id"]),
            )
        )
    return refs


# =========================
# Detector: context + kick spike
# =========================
def detect_restart_frame_with_context(
    data_frames: list[dict],
    ball_df: pd.DataFrame,
    period: int,
    t0_sec: float,
    pressing_team_id: int = PRESSING_TEAM_ID,
    pre_s: float = PRE_S,
    post_s: float = POST_S,
    lookahead_frames: int = LOOKAHEAD_FRAMES,
) -> int | None:

    w = ball_df[
        (ball_df["period"] == period) &
        (ball_df["ts_sec"] >= t0_sec - pre_s) &
        (ball_df["ts_sec"] <= t0_sec + post_s)
    ][["frame","ts_sec","x_i","y_i","z_i","kick_candidate"]].copy()

    if w.empty:
        return None

    # infer goal side (left/right) for this event
    def_side = infer_defending_side_from_ball_window(w, pitch_length=PITCH_LENGTH)

    # frame lookup for this period
    frame_map = {fr.get("frame"): fr for fr in data_frames if fr.get("period") == period}

    def scan(area_fn):
        for _, r in w.iterrows():
            fr = int(r["frame"])
            fr_dict = frame_map.get(fr)
            if fr_dict is None:
                continue

            x, y = r["x_i"], r["y_i"]
            if not area_fn(float(x), float(y), defending_side=def_side):
                continue

            # pressing team must be outside penalty area
            if count_pressing_team_in_pen_area(fr_dict, pressing_team_id, defending_side=def_side) > 0:
                continue

            # find kick_candidate shortly after
            w2 = w[w["frame"].between(fr, fr + lookahead_frames)]
            if MAX_Z_AT_KICK is not None:
                w2 = w2[w2["z_i"].isna() | (w2["z_i"] <= MAX_Z_AT_KICK)]

            if (w2["kick_candidate"] == True).any():
                return int(w2.loc[w2["kick_candidate"] == True, "frame"].iloc[0])

        return None

    # 1) goal area
    rf = scan(in_goal_area_buf)
    if rf is not None:
        return rf

    # 2) fallback penalty area
    rf = scan(in_penalty_area_buf)
    if rf is not None:
        return rf

    # 3) fallback: first kick_candidate after t0
    w_after = w[w["ts_sec"] >= t0_sec]
    if (w_after["kick_candidate"] == True).any():
        return int(w_after.loc[w_after["kick_candidate"] == True, "frame"].iloc[0])

    return None


# =========================
# Diagnostics
# =========================
def plot_goal_kick_diagnostic(ball_df: pd.DataFrame, gk_df: pd.DataFrame, idx: int = 0, window_s: int = 8):
    gk_valid = gk_df.dropna(subset=["restart_frame"]).reset_index(drop=True)
    if gk_valid.empty:
        print("No restart_frame to plot.")
        return

    row = gk_valid.iloc[idx]
    period = int(row["period"])
    rf = int(row["restart_frame"])

    w = ball_df[
        (ball_df["period"] == period) &
        (ball_df["frame"].between(rf - window_s*FPS, rf + window_s*FPS))
    ][["frame","ts_sec","speed","acc","kick_candidate"]].copy()

    if w.empty:
        print("Empty window.")
        return

    t_rf = ball_df.loc[(ball_df["period"] == period) & (ball_df["frame"] == rf), "ts_sec"].iloc[0]

    plt.figure()
    plt.plot(w["ts_sec"], w["speed"])
    plt.axvline(t_rf)
    plt.title(f"Speed | idx={idx} | period={period} | restart_frame={rf}")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.show()

    plt.figure()
    plt.plot(w["ts_sec"], w["acc"])
    plt.axvline(t_rf)
    plt.title(f"Acceleration | idx={idx} | period={period} | restart_frame={rf}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.show()

    print(row[["event_id","time_start","t0_sec","restart_frame"]])
    print(w.tail(20))


def diagnose_one(ref: GoalKickRef, ball_df: pd.DataFrame, tracking_data: list[dict]) -> dict:
    period = ref.period
    t0 = parse_mmss_to_seconds(ref.time_start)

    w = ball_df[
        (ball_df["period"] == period) &
        (ball_df["ts_sec"] >= t0 - PRE_S) &
        (ball_df["ts_sec"] <= t0 + POST_S)
    ][["frame","x_i","y_i","z_i","kick_candidate"]].copy()

    if w.empty:
        return {"event_id": ref.event_id, "n": 0}

    side = infer_defending_side_from_ball_window(w, pitch_length=PITCH_LENGTH)

    frame_map = {fr.get("frame"): fr for fr in tracking_data if fr.get("period") == period}

    c1 = w.apply(lambda r: in_goal_area_buf(r["x_i"], r["y_i"], side), axis=1)
    n1 = int(c1.sum())

    def c2_ok(frame):
        fr_dict = frame_map.get(int(frame))
        if fr_dict is None:
            return False
        return count_pressing_team_in_pen_area(fr_dict, PRESSING_TEAM_ID, side) == 0

    c2 = w["frame"].map(c2_ok)
    n2 = int(c2.sum())

    n12 = int((c1 & c2).sum())
    n3 = int((w["kick_candidate"] == True).sum())

    return {
        "event_id": ref.event_id,
        "period": period,
        "n": len(w),
        "side_inferred": side,
        "ball_in_goal_area_buf": n1,
        "real_out_pen_area": n2,
        "both": n12,
        "kick_candidates": n3,
    }


# =========================
# MAIN
# =========================
def main():
    # --- load data ---
    with open(TRACKING_PATH, "r") as f:
        tracking_data = json.load(f)

    dynamic_df = pd.read_parquet(DYNAMIC_PATH)

    # meta not strictly needed anymore, but keep it loaded
    with open(META_PATH, "r") as f:
        _meta_data = json.load(f)

    # --- build ball_df base ---
    rows = []
    for fr in tracking_data:
        ball = fr.get("ball_data") or {}
        rows.append({
            "frame": fr.get("frame"),
            "timestamp": fr.get("timestamp"),
            "period": fr.get("period"),
            "x": ball.get("x"),
            "y": ball.get("y"),
            "z": ball.get("z"),
            "is_detected": ball.get("is_detected"),
        })

    ball_df = pd.DataFrame(rows).sort_values(["period", "frame"]).reset_index(drop=True)

    for c in ["x", "y", "z", "is_detected"]:
        ball_df[c] = pd.to_numeric(ball_df[c], errors="coerce")

    ball_df["ts_sec"] = ball_df["timestamp"].map(parse_hhmmss_to_seconds)

    # --- interpolate + kinematics + kick_candidate ---
    ball_df["x_i"] = ball_df.groupby("period")["x"].transform(
        lambda s: s.interpolate(limit=MAX_GAP, limit_direction="both")
    )
    ball_df["y_i"] = ball_df.groupby("period")["y"].transform(
        lambda s: s.interpolate(limit=MAX_GAP, limit_direction="both")
    )
    ball_df["z_i"] = ball_df.groupby("period")["z"].transform(
        lambda s: s.interpolate(limit=MAX_GAP, limit_direction="both")
    )

    ball_df["ball_valid"] = ball_df["x_i"].notna() & ball_df["y_i"].notna()

    ball_df["vx"] = ball_df.groupby("period")["x_i"].diff() / DT
    ball_df["vy"] = ball_df.groupby("period")["y_i"].diff() / DT
    ball_df["speed"] = np.sqrt(ball_df["vx"]**2 + ball_df["vy"]**2)

    ball_df["ax"] = ball_df.groupby("period")["vx"].diff() / DT
    ball_df["ay"] = ball_df.groupby("period")["vy"].diff() / DT
    ball_df["acc"] = np.sqrt(ball_df["ax"]**2 + ball_df["ay"]**2)

    ball_df.loc[~ball_df["ball_valid"], ["vx","vy","speed","ax","ay","acc"]] = np.nan

    ball_df["kick_candidate"] = (ball_df["speed"] > KICK_SPEED) & (ball_df["acc"] > KICK_ACC)

    # --- refs from dynamic ---
    refs = filter_goal_kick_refs(dynamic_df, team_id=HOME_TEAM_ID, game_id=game_id)
    print("Goal kicks (home team refs):", len(refs))

    # --- detect restarts ---
    results = []
    for ref in refs:
        t0 = parse_mmss_to_seconds(ref.time_start)
        rf = detect_restart_frame_with_context(
            data_frames=tracking_data,
            ball_df=ball_df,
            period=ref.period,
            t0_sec=t0
        )
        results.append({
            "game_id": ref.game_id,
            "period": ref.period,
            "event_id": ref.event_id,
            "time_start": ref.time_start,
            "t0_sec": t0,
            "restart_frame": rf
        })

    gk_starts = pd.DataFrame(results)

    print(gk_starts.head())
    print("Context restart detection rate:", gk_starts["restart_frame"].notna().mean())

    # --- validate visually ---
    plot_goal_kick_diagnostic(ball_df, gk_starts, idx=0, window_s=8)
    plot_goal_kick_diagnostic(ball_df, gk_starts, idx=1, window_s=8)

# --- diagnose first 5 refs ---
    for ref in refs[:5]:
        print(diagnose_one(ref, ball_df, tracking_data))

    return gk_starts   # <- TEM QUE ESTAR AQUI DENTRO


if __name__ == "__main__":
    gk_starts = main()
print("Final detection rate:", gk_starts["restart_frame"].notna().mean())
print(gk_starts.head())
    
gk_starts.head()


