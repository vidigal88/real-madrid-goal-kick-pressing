"""Build-up phase detection for goal kick scenarios.

This module contains the core logic for detecting and analyzing build-up phases
from tracking data, including setup detection, ready state identification,
and kick moment detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .geometry import ball_in_goal_area, distance, goal_area_x_bounds
from .goal_kick_detector import GoalKickRef
from .team_utils import find_starting_goalkeeper
from .time_utils import dynamic_to_tracking_time, time_to_seconds, seconds_to_time


@dataclass(frozen=True)
class BuildUp:
    """Complete build-up phase information for a goal kick.

    Attributes:
        game_id: Match identifier
        period: Match period
        opponent_team_id: ID of team taking the goal kick
        opponent_team_name: Name of team taking the goal kick
        rm_team_id: Real Madrid team ID
        rm_team_name: Real Madrid team name
        gk_id: Goalkeeper player ID
        gk_name: Goalkeeper name
        gk_side: Side of pitch goalkeeper is defending ("left" or "right")
        event_id: Goal kick event identifier
        event_reference_time: Dynamic time of the goal kick event
        setup_start_time: When setup phase begins (tracking time)
        setup_end_time: When setup phase ends (tracking time)
        ready_time: When goalkeeper is ready to kick (tracking time)
        kick_time: When ball is kicked (tracking time)
        time_before_event_s: Seconds between kick and event reference
    """
    game_id: str
    period: int
    opponent_team_id: int
    opponent_team_name: str
    rm_team_id: int
    rm_team_name: str
    gk_id: int
    gk_name: str
    gk_side: str  # "left" or "right"
    event_id: str
    event_reference_time: str  # dynamic time
    setup_start_time: str  # tracking time "HH:MM:SS.DD"
    setup_end_time: str  # tracking time
    ready_time: str  # tracking time
    kick_time: str  # tracking time
    time_before_event_s: float


@dataclass(frozen=True)
class GoalKickCandidate:
    """Candidate kick frame that satisfies all hard scene filters."""
    time: str
    frame: int
    gk_dist: float
    speed: float
    acc: float


FPS = 25.0
DT = 1.0 / FPS
KICK_SPEED_MPS = 5.0
KICK_ACC_MPS2 = 25.0
KICK_NEIGHBORHOOD_FRAMES = 2
GOAL_AREA_NEIGHBORHOOD_FRAMES = 1
GK_CONTEXT_DISTANCE_M = 15.0
BALL_GOAL_AREA_MARGIN_M = 2.5
LAUNCH_ZONE_X_MARGIN_M = 6.0
LAUNCH_ZONE_Y_MARGIN_M = 4.0
LAUNCH_ZONE_GK_CONTEXT_DISTANCE_M = 10.0
RM_GOAL_AREA_MARGIN_M = -0.5
KICK_MAX_DISPLACEMENT_M = 1.0
KICK_MIN_FUTURE_FRAMES = 2
RM_GOAL_AREA_BLOCK_MIN_FRAMES = 2


def merge_ball_and_gk(
    tracking_df: pd.DataFrame,
    period: int,
    start_time: str,
    end_time: str,
    gk_id: int
) -> pd.DataFrame:
    """Merge ball and goalkeeper positions for a time window.

    Args:
        tracking_df: Full tracking DataFrame
        period: Match period
        start_time: Window start time (tracking format)
        end_time: Window end time (tracking format)
        gk_id: Goalkeeper player ID

    Returns:
        DataFrame with columns: time, frame, ball_x, ball_y, gk_x, gk_y
        Only includes frames where both ball and goalkeeper are detected
    """
    df = tracking_df[(tracking_df["period"] == period) & (tracking_df["time"] >= start_time) & (tracking_df["time"] <= end_time)]
    df = df[df["is_detected"] == True][["time", "frame", "player_id", "is_ball", "x", "y"]].copy()

    ball = df[df["is_ball"] == True][["time", "frame", "x", "y"]].rename(columns={"x": "ball_x", "y": "ball_y"})
    gk = df[(df["is_ball"] == False) & (df["player_id"] == gk_id)][["time", "frame", "x", "y"]].rename(
        columns={"x": "gk_x", "y": "gk_y"}
    )
    return pd.merge(ball, gk, on=["time", "frame"], how="inner").sort_values(["frame"])


def add_ball_kinematics(merged: pd.DataFrame) -> pd.DataFrame:
    """Add simple ball kinematics to a merged ball/GK frame table."""
    if merged.empty:
        return merged.copy()

    out = merged.sort_values(["frame"]).copy()
    out["vx"] = out["ball_x"].astype(float).diff() / DT
    out["vy"] = out["ball_y"].astype(float).diff() / DT
    out["speed"] = np.sqrt(out["vx"] ** 2 + out["vy"] ** 2)
    out["ax"] = out["vx"].diff() / DT
    out["ay"] = out["vy"].diff() / DT
    out["acc"] = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2)
    return out


def count_team_players_in_goal_area(
    tracking_df: pd.DataFrame,
    *,
    period: int,
    frame: int,
    player_ids: set[int],
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
    margin_m: float = 0.0,
) -> int:
    """Count detected outfield/team players inside the opponent goal area for one frame."""
    if not player_ids:
        return 0

    df = tracking_df[
        (tracking_df["period"] == int(period))
        & (tracking_df["frame"] == int(frame))
        & (tracking_df["is_ball"] == False)
        & (tracking_df["is_detected"] == True)
        & (tracking_df["player_id"].isin(player_ids))
    ][["x", "y"]]

    if df.empty:
        return 0

    return int(
        df.apply(
            lambda r: ball_in_goal_area(
                float(r["x"]),
                float(r["y"]),
                goal_x_min - float(margin_m),
                goal_x_max + float(margin_m),
                goal_area_half_width_m + float(margin_m),
            ),
            axis=1,
        ).sum()
    )


def min_gk_ball_distance_around_frame(
    merged: pd.DataFrame,
    *,
    frame: int,
    neighborhood_frames: int,
) -> float | None:
    """Return the minimum GK-ball distance in a small frame neighborhood."""
    w = merged[merged["frame"].between(int(frame) - int(neighborhood_frames), int(frame) + int(neighborhood_frames))].copy()
    if w.empty:
        return None

    dists = w.apply(
        lambda r: distance(float(r["ball_x"]), float(r["ball_y"]), float(r["gk_x"]), float(r["gk_y"])),
        axis=1,
    )
    if dists.empty:
        return None
    return float(dists.min())


def ball_in_effective_goal_area(
    ball_x: float,
    ball_y: float,
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
) -> bool:
    """Allow a small spatial tolerance because tracking can jitter around the six-yard box."""
    return ball_in_goal_area(
        ball_x,
        ball_y,
        goal_x_min - BALL_GOAL_AREA_MARGIN_M,
        goal_x_max + BALL_GOAL_AREA_MARGIN_M,
        goal_area_half_width_m + BALL_GOAL_AREA_MARGIN_M,
    )


def ball_in_goal_kick_launch_zone(
    ball_x: float,
    ball_y: float,
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
) -> bool:
    """Controlled fallback for noisy tracking just outside the six-yard box."""
    return ball_in_goal_area(
        ball_x,
        ball_y,
        goal_x_min - LAUNCH_ZONE_X_MARGIN_M,
        goal_x_max + LAUNCH_ZONE_X_MARGIN_M,
        goal_area_half_width_m + LAUNCH_ZONE_Y_MARGIN_M,
    )


def count_rm_goal_area_presence_in_window(
    tracking_df: pd.DataFrame,
    *,
    period: int,
    frame: int,
    player_ids: set[int],
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
    neighborhood_frames: int,
) -> int:
    """Count how many nearby frames show RM players inside the goal area."""
    presence = 0
    for neighbor in range(int(frame) - int(neighborhood_frames), int(frame) + int(neighborhood_frames) + 1):
        if count_team_players_in_goal_area(
            tracking_df,
            period=period,
            frame=neighbor,
            player_ids=player_ids,
            goal_x_min=goal_x_min,
            goal_x_max=goal_x_max,
            goal_area_half_width_m=goal_area_half_width_m,
            margin_m=RM_GOAL_AREA_MARGIN_M,
        ) > 0:
            presence += 1
    return presence


def evaluate_goal_kick_window(
    *,
    tracking_df: pd.DataFrame,
    merged: pd.DataFrame,
    period: int,
    rm_player_ids: set[int],
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
    gk_ball_distance_m: float,
    kick_displacement_m: float,
    confirm_frames: int,
) -> dict[str, Any]:
    """Evaluate all candidate stages for one goal-kick reference window."""
    diag: dict[str, Any] = {
        "period": int(period),
        "n_window_frames": 0,
        "n_goal_area_frames": 0,
        "n_kick_like_frames": 0,
        "n_goal_area_kick_like_frames": 0,
        "n_rm_clear_candidates": 0,
        "n_gk_near_candidates": 0,
        "n_displacement_candidates": 0,
        "failure_reason": None,
        "selected_kick_time": None,
        "selected_kick_frame": None,
        "selected_gk_dist": np.nan,
        "selected_speed": np.nan,
        "selected_acc": np.nan,
    }

    if merged.empty:
        diag["failure_reason"] = "empty_merge"
        return diag

    w = add_ball_kinematics(merged)
    if w.empty:
        diag["failure_reason"] = "empty_kinematics"
        return diag

    diag["n_window_frames"] = int(len(w))

    w["gk_ball_dist"] = w.apply(
        lambda r: distance(float(r["ball_x"]), float(r["ball_y"]), float(r["gk_x"]), float(r["gk_y"])),
        axis=1,
    )
    w["ball_in_goal_area"] = w.apply(
        lambda r: ball_in_effective_goal_area(
            float(r["ball_x"]),
            float(r["ball_y"]),
            goal_x_min,
            goal_x_max,
            goal_area_half_width_m,
        ),
        axis=1,
    )
    w["ball_in_launch_zone"] = w.apply(
        lambda r: ball_in_goal_kick_launch_zone(
            float(r["ball_x"]),
            float(r["ball_y"]),
            goal_x_min,
            goal_x_max,
            goal_area_half_width_m,
        ),
        axis=1,
    )
    w["kick_like"] = (w["speed"] >= KICK_SPEED_MPS) | (w["acc"] >= KICK_ACC_MPS2)

    ball_goal_frames = set(w.loc[w["ball_in_goal_area"], "frame"].astype(int).tolist())
    ball_launch_frames = set(w.loc[w["ball_in_launch_zone"], "frame"].astype(int).tolist())
    w["ball_in_goal_area_nearby"] = w["frame"].map(
        lambda fr: any(
            int(fr) + offset in ball_goal_frames
            for offset in range(-GOAL_AREA_NEIGHBORHOOD_FRAMES, GOAL_AREA_NEIGHBORHOOD_FRAMES + 1)
        )
    )
    w["ball_in_launch_zone_nearby"] = w["frame"].map(
        lambda fr: any(
            int(fr) + offset in ball_launch_frames
            for offset in range(-GOAL_AREA_NEIGHBORHOOD_FRAMES, GOAL_AREA_NEIGHBORHOOD_FRAMES + 1)
        )
    )

    diag["n_goal_area_frames"] = int(w["ball_in_goal_area_nearby"].sum())
    diag["n_kick_like_frames"] = int(w["kick_like"].sum())
    diag["n_goal_area_kick_like_frames"] = int((w["ball_in_goal_area_nearby"] & w["kick_like"]).sum())

    spatial_ok = w["ball_in_goal_area_nearby"] | w["ball_in_launch_zone_nearby"]

    if not bool(spatial_ok.any()):
        diag["failure_reason"] = "no_goal_area_frame"
        return diag
    if not bool((spatial_ok & w["kick_like"]).any()):
        diag["failure_reason"] = "no_goal_area_kicklike"
        return diag

    candidates: list[GoalKickCandidate] = []
    for _, row in w.iterrows():
        if not bool(row["kick_like"]):
            continue
        if not bool(row["ball_in_goal_area_nearby"] or row["ball_in_launch_zone_nearby"]):
            continue

        frame = int(row["frame"])
        time = str(row["time"])

        rm_presence_frames = count_rm_goal_area_presence_in_window(
            tracking_df,
            period=period,
            frame=frame,
            player_ids=rm_player_ids,
            goal_x_min=goal_x_min,
            goal_x_max=goal_x_max,
            goal_area_half_width_m=goal_area_half_width_m,
            neighborhood_frames=GOAL_AREA_NEIGHBORHOOD_FRAMES,
        )
        if rm_presence_frames >= RM_GOAL_AREA_BLOCK_MIN_FRAMES:
            continue
        diag["n_rm_clear_candidates"] += 1

        gk_near_dist = min_gk_ball_distance_around_frame(
            w,
            frame=frame,
            neighborhood_frames=KICK_NEIGHBORHOOD_FRAMES,
        )
        gk_limit = max(float(gk_ball_distance_m), GK_CONTEXT_DISTANCE_M)
        if not bool(row["ball_in_goal_area_nearby"]) and bool(row["ball_in_launch_zone_nearby"]):
            gk_limit = min(gk_limit, LAUNCH_ZONE_GK_CONTEXT_DISTANCE_M)
        if gk_near_dist is None or gk_near_dist > gk_limit:
            continue
        diag["n_gk_near_candidates"] += 1

        future = w[w["frame"].between(frame + 1, frame + int(confirm_frames))].copy()
        if len(future) < min(int(confirm_frames), KICK_MIN_FUTURE_FRAMES):
            continue

        displacement = future.apply(
            lambda r: distance(float(r["ball_x"]), float(r["ball_y"]), float(row["ball_x"]), float(row["ball_y"])),
            axis=1,
        )
        if float(displacement.max()) < max(float(kick_displacement_m), KICK_MAX_DISPLACEMENT_M):
            continue
        diag["n_displacement_candidates"] += 1

        candidates.append(
            GoalKickCandidate(
                time=time,
                frame=frame,
                gk_dist=float(gk_near_dist),
                speed=float(row["speed"]),
                acc=float(row["acc"]),
            )
        )

    if diag["n_rm_clear_candidates"] == 0:
        diag["failure_reason"] = "rm_in_goal_area_blocks_all"
        return diag
    if diag["n_gk_near_candidates"] == 0:
        diag["failure_reason"] = "gk_context_blocks_all"
        return diag
    if diag["n_displacement_candidates"] == 0:
        diag["failure_reason"] = "displacement_blocks_all"
        return diag

    candidates.sort(key=lambda c: (-c.frame, c.gk_dist, -c.speed, -c.acc))
    best = candidates[0]
    diag["selected_kick_time"] = str(best.time)
    diag["selected_kick_frame"] = int(best.frame)
    diag["selected_gk_dist"] = float(best.gk_dist)
    diag["selected_speed"] = float(best.speed)
    diag["selected_acc"] = float(best.acc)
    return diag


def find_setup_segments(
    merged: pd.DataFrame,
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
    gk_ball_distance_m: float,
    min_frames: int,
    gap_frames: int,
) -> list[pd.DataFrame]:
    """Find continuous segments where ball is in goal area near goalkeeper.

    A valid setup segment is a continuous sequence of frames where:
    - Ball is within the goal area bounds
    - Ball is within specified distance of goalkeeper
    - Segment lasts at least min_frames
    - Allows brief interruptions up to gap_frames

    Args:
        merged: DataFrame with ball_x, ball_y, gk_x, gk_y, time, frame
        goal_x_min: Minimum X coordinate of goal area
        goal_x_max: Maximum X coordinate of goal area
        goal_area_half_width_m: Half-width of goal area (Y-axis)
        gk_ball_distance_m: Maximum distance between GK and ball
        min_frames: Minimum frames for valid segment
        gap_frames: Maximum gap frames to tolerate

    Returns:
        List of DataFrames, each representing a valid setup segment
    """
    if merged.empty:
        return []

    merged = merged.copy()
    merged["ok"] = (
        merged.apply(
            lambda r: ball_in_goal_area(float(r["ball_x"]), float(r["ball_y"]), goal_x_min, goal_x_max, goal_area_half_width_m)
            and distance(float(r["ball_x"]), float(r["ball_y"]), float(r["gk_x"]), float(r["gk_y"])) <= gk_ball_distance_m,
            axis=1,
        )
    )

    segments: list[pd.DataFrame] = []
    current_rows: list[dict[str, Any]] = []
    gap = 0
    last_frame: int | None = None

    for r in merged.itertuples(index=False):
        frame = int(r.frame)
        if last_frame is not None:
            missed = frame - last_frame - 1
            if missed > 0 and current_rows:
                gap += missed
        last_frame = frame

        if bool(r.ok):
            current_rows.append(r._asdict())
            gap = 0
            continue

        if current_rows:
            gap += 1
            if gap > gap_frames:
                if len(current_rows) >= min_frames:
                    segments.append(pd.DataFrame(current_rows))
                current_rows = []
                gap = 0

    if current_rows and len(current_rows) >= min_frames:
        segments.append(pd.DataFrame(current_rows))
    return segments


def detect_goal_kick_candidate_from_window(
    *,
    tracking_df: pd.DataFrame,
    merged: pd.DataFrame,
    period: int,
    rm_player_ids: set[int],
    goal_x_min: float,
    goal_x_max: float,
    goal_area_half_width_m: float,
    gk_ball_distance_m: float,
    kick_displacement_m: float,
    confirm_frames: int,
    debug: list[str],
) -> tuple[str, int] | None:
    """Detect the best goal-kick candidate frame within a pre-event window.

    Hard filters at the kick frame:
    - ball is inside the goal area
    - no Real Madrid player is inside the same goal area
    - ball motion looks like a kick via speed/acceleration
    - ball movement is sustained after the candidate
    - opponent starting goalkeeper is close to the ball around that frame
    """
    diag = evaluate_goal_kick_window(
        tracking_df=tracking_df,
        merged=merged,
        period=period,
        rm_player_ids=rm_player_ids,
        goal_x_min=goal_x_min,
        goal_x_max=goal_x_max,
        goal_area_half_width_m=goal_area_half_width_m,
        gk_ball_distance_m=gk_ball_distance_m,
        kick_displacement_m=kick_displacement_m,
        confirm_frames=confirm_frames,
    )
    if not diag["selected_kick_time"] or diag["selected_kick_frame"] is None:
        reason = str(diag.get("failure_reason") or "no_candidate")
        if reason == "no_goal_area_frame":
            debug.append(f"[P{period}] No frames with ball in goal area before event")
        else:
            debug.append(f"[P{period}] {reason}")
        return None
    return str(diag["selected_kick_time"]), int(diag["selected_kick_frame"])


def find_ready_time_in_setup(
    seg: pd.DataFrame,
    *,
    ready_min_dist_m: float,
    ready_max_dist_m: float,
    stable_frames: int,
    ball_step_eps_m: float,
    gk_step_eps_m: float,
) -> str | None:
    """Find the moment when goalkeeper is ready to kick within a setup segment.

    Ready state is defined as:
    - GK-ball distance within [ready_min_dist_m, ready_max_dist_m]
    - Ball movement per frame <= ball_step_eps_m
    - GK movement per frame <= gk_step_eps_m
    - All conditions sustained for stable_frames consecutive frames

    Args:
        seg: Setup segment DataFrame with ball_x, ball_y, gk_x, gk_y, time, frame
        ready_min_dist_m: Minimum GK-ball distance for ready state
        ready_max_dist_m: Maximum GK-ball distance for ready state
        stable_frames: Consecutive frames needed for ready state
        ball_step_eps_m: Maximum ball movement per frame
        gk_step_eps_m: Maximum GK movement per frame

    Returns:
        Time when ready state begins (tracking format), None if not found
    """
    if seg.empty or len(seg) < max(2, stable_frames):
        return None

    s = seg.sort_values(["frame"]).copy()
    s["dist"] = s.apply(
        lambda r: distance(float(r["ball_x"]), float(r["ball_y"]), float(r["gk_x"]), float(r["gk_y"])), axis=1
    )
    s["ball_step"] = (
        (s["ball_x"].astype(float).diff() ** 2 + s["ball_y"].astype(float).diff() ** 2).pow(0.5).fillna(0.0)
    )
    s["gk_step"] = ((s["gk_x"].astype(float).diff() ** 2 + s["gk_y"].astype(float).diff() ** 2).pow(0.5).fillna(0.0))

    ok = (
        (s["dist"] >= float(ready_min_dist_m))
        & (s["dist"] <= float(ready_max_dist_m))
        & (s["ball_step"] <= float(ball_step_eps_m))
        & (s["gk_step"] <= float(gk_step_eps_m))
    )

    consec = 0
    best_start_idx: int | None = None
    for i, v in enumerate(ok.tolist()):
        if v:
            consec += 1
            if consec >= int(stable_frames):
                best_start_idx = i - int(stable_frames) + 1
        else:
            consec = 0

    if best_start_idx is None:
        return None
    return str(s["time"].iloc[int(best_start_idx)])


def detect_build_up_from_reference(
    *,
    tracking_df: pd.DataFrame,
    meta: dict[str, Any],
    goal_kick_ref: GoalKickRef,
    rm_team: dict[str, Any],
    opponent_team: dict[str, Any],
    lookback_seconds: int,
    goal_area_depth_m: float,
    goal_area_half_width_m: float,
    goal_area_x_margin_m: float,
    gk_ball_distance_m: float,
    kick_displacement_m: float,
    kick_confirm_frames: int,
    ready_min_dist_m: float,
    ready_max_dist_m: float,
    ready_stable_frames: int,
    ready_ball_step_eps_m: float,
    ready_gk_step_eps_m: float,
    min_setup_frames: int,
    setup_gap_frames: int,
    debug: list[str],
) -> BuildUp | None:
    """Detect complete build-up phase from a goal kick reference event.

    This is the main detection function that:
    1. Finds the opponent's starting goalkeeper
    2. Determines which side they're defending
    3. Searches for setup segments where ball is in goal area near GK
    4. Identifies ready state within setup
    5. Detects kick moment after setup
    6. Returns complete BuildUp object if successful

    Args:
        tracking_df: Full tracking DataFrame
        meta: Match metadata
        goal_kick_ref: Goal kick reference from dynamic data
        rm_team: Real Madrid team dictionary
        opponent_team: Opponent team dictionary
        lookback_seconds: How far back to search before event
        goal_area_depth_m: Depth of goal area from goal line
        goal_area_half_width_m: Half-width of goal area
        goal_area_x_margin_m: Extra margin for goal area
        gk_ball_distance_m: Max distance between GK and ball for setup
        kick_displacement_m: Min ball displacement to detect kick
        kick_confirm_frames: Frames to confirm sustained kick
        ready_min_dist_m: Min GK-ball distance for ready state
        ready_max_dist_m: Max GK-ball distance for ready state
        ready_stable_frames: Frames needed for stable ready state
        ready_ball_step_eps_m: Max ball movement for ready state
        ready_gk_step_eps_m: Max GK movement for ready state
        min_setup_frames: Minimum frames for valid setup segment
        setup_gap_frames: Max gap frames in setup segment
        debug: List to append debug messages to

    Returns:
        BuildUp object if detection successful, None otherwise
    """
    from .goal_kick_detector import detect_goalkeeper_side_from_parquet

    gk = find_starting_goalkeeper(meta, int(opponent_team["id"]))
    if not gk:
        debug.append(f"[{goal_kick_ref.game_id}] No starting GK for opponent team_id={opponent_team['id']}")
        return None

    gk_id = int(gk["id"])
    gk_name = str(gk.get("short_name") or "GK")
    period = int(goal_kick_ref.period)

    gk_side = detect_goalkeeper_side_from_parquet(tracking_df, gk_id, period)
    if not gk_side:
        debug.append(f"[{goal_kick_ref.game_id}] Could not detect GK side for GK={gk_name} in period={period}")
        return None

    pitch_length = float(meta.get("pitch_length") or 105.0)
    half_length = pitch_length / 2.0
    goal_x_min, goal_x_max = goal_area_x_bounds(half_length, goal_area_depth_m, goal_area_x_margin_m, gk_side)
    rm_player_ids = {
        int(p["id"])
        for p in meta.get("players", [])
        if p.get("team_id") == int(rm_team["id"]) and p.get("id") is not None
    }

    event_time_tracking = dynamic_to_tracking_time(goal_kick_ref.time_start, period=period)
    event_sec = time_to_seconds(event_time_tracking)
    lookback_start = seconds_to_time(max(0.0, event_sec - float(lookback_seconds)))

    merged = merge_ball_and_gk(tracking_df, period, lookback_start, event_time_tracking, gk_id)
    if merged.empty:
        return None

    candidate = detect_goal_kick_candidate_from_window(
        tracking_df=tracking_df,
        merged=merged,
        period=period,
        rm_player_ids=rm_player_ids,
        goal_x_min=goal_x_min,
        goal_x_max=goal_x_max,
        goal_area_half_width_m=goal_area_half_width_m,
        gk_ball_distance_m=gk_ball_distance_m,
        kick_displacement_m=kick_displacement_m,
        confirm_frames=int(kick_confirm_frames),
        debug=debug,
    )
    if not candidate:
        return None

    kick_time, kick_frame = candidate

    pre_kick = merged[merged["frame"] <= int(kick_frame)].copy()
    segments = find_setup_segments(
        pre_kick,
        goal_x_min,
        goal_x_max,
        goal_area_half_width_m,
        gk_ball_distance_m,
        min_setup_frames,
        setup_gap_frames,
    )
    if not segments:
        setup_start = str(kick_time)
        setup_end = str(kick_time)
        ready_time = str(kick_time)
        debug.append(f"[{goal_kick_ref.game_id}] kick={kick_time} with no setup segment; using kick frame as setup fallback")
    else:
        seg = segments[-1]
        setup_start = str(seg["time"].iloc[0])
        setup_end = str(seg["time"].iloc[-1])

        ready_time = find_ready_time_in_setup(
            seg,
            ready_min_dist_m=ready_min_dist_m,
            ready_max_dist_m=ready_max_dist_m,
            stable_frames=ready_stable_frames,
            ball_step_eps_m=ready_ball_step_eps_m,
            gk_step_eps_m=ready_gk_step_eps_m,
        )

    time_before_event = time_to_seconds(event_time_tracking) - time_to_seconds(kick_time)
    debug.append(
        f"[{goal_kick_ref.game_id}] ref={goal_kick_ref.time_start} (P{period}) setup={setup_start}->{setup_end} kick={kick_time} Δevent={time_before_event:.1f}s"
    )

    return BuildUp(
        game_id=goal_kick_ref.game_id,
        period=period,
        opponent_team_id=int(opponent_team["id"]),
        opponent_team_name=str(opponent_team["name"]),
        rm_team_id=int(rm_team["id"]),
        rm_team_name=str(rm_team["name"]),
        gk_id=gk_id,
        gk_name=gk_name,
        gk_side=gk_side,
        event_id=goal_kick_ref.event_id,
        event_reference_time=goal_kick_ref.time_start,
        setup_start_time=setup_start,
        setup_end_time=setup_end,
        ready_time=str(ready_time or setup_end),
        kick_time=kick_time,
        time_before_event_s=float(time_before_event),
    )


def diagnose_build_up_from_reference(
    *,
    tracking_df: pd.DataFrame,
    meta: dict[str, Any],
    goal_kick_ref: GoalKickRef,
    rm_team: dict[str, Any],
    opponent_team: dict[str, Any],
    lookback_seconds: int,
    goal_area_depth_m: float,
    goal_area_half_width_m: float,
    goal_area_x_margin_m: float,
    gk_ball_distance_m: float,
    kick_displacement_m: float,
    kick_confirm_frames: int,
    ready_min_dist_m: float,
    ready_max_dist_m: float,
    ready_stable_frames: int,
    ready_ball_step_eps_m: float,
    ready_gk_step_eps_m: float,
    min_setup_frames: int,
    setup_gap_frames: int,
) -> dict[str, Any]:
    """Return per-reference diagnostics for goal-kick build-up detection."""
    from .goal_kick_detector import detect_goalkeeper_side_from_parquet

    diag: dict[str, Any] = {
        "game_id": str(goal_kick_ref.game_id),
        "period": int(goal_kick_ref.period),
        "event_id": str(goal_kick_ref.event_id),
        "event_reference_time": str(goal_kick_ref.time_start),
        "rm_team_id": int(rm_team["id"]),
        "opponent_team_id": int(opponent_team["id"]),
        "gk_id": pd.NA,
        "gk_name": pd.NA,
        "gk_side": pd.NA,
        "failure_reason": None,
        "selected_kick_time": pd.NA,
        "selected_kick_frame": pd.NA,
        "selected_gk_dist": np.nan,
        "selected_speed": np.nan,
        "selected_acc": np.nan,
        "setup_found": False,
        "setup_start_time": pd.NA,
        "setup_end_time": pd.NA,
        "ready_time": pd.NA,
        "time_before_event_s": np.nan,
        "build_up_detected": False,
    }

    gk = find_starting_goalkeeper(meta, int(opponent_team["id"]))
    if not gk:
        diag["failure_reason"] = "no_starting_gk"
        return diag

    gk_id = int(gk["id"])
    gk_name = str(gk.get("short_name") or "GK")
    period = int(goal_kick_ref.period)
    diag["gk_id"] = gk_id
    diag["gk_name"] = gk_name

    gk_side = detect_goalkeeper_side_from_parquet(tracking_df, gk_id, period)
    if not gk_side:
        diag["failure_reason"] = "no_gk_side"
        return diag
    diag["gk_side"] = gk_side

    pitch_length = float(meta.get("pitch_length") or 105.0)
    half_length = pitch_length / 2.0
    goal_x_min, goal_x_max = goal_area_x_bounds(half_length, goal_area_depth_m, goal_area_x_margin_m, gk_side)
    rm_player_ids = {
        int(p["id"])
        for p in meta.get("players", [])
        if p.get("team_id") == int(rm_team["id"]) and p.get("id") is not None
    }

    event_time_tracking = dynamic_to_tracking_time(goal_kick_ref.time_start, period=period)
    event_sec = time_to_seconds(event_time_tracking)
    lookback_start = seconds_to_time(max(0.0, event_sec - float(lookback_seconds)))

    merged = merge_ball_and_gk(tracking_df, period, lookback_start, event_time_tracking, gk_id)
    eval_diag = evaluate_goal_kick_window(
        tracking_df=tracking_df,
        merged=merged,
        period=period,
        rm_player_ids=rm_player_ids,
        goal_x_min=goal_x_min,
        goal_x_max=goal_x_max,
        goal_area_half_width_m=goal_area_half_width_m,
        gk_ball_distance_m=gk_ball_distance_m,
        kick_displacement_m=kick_displacement_m,
        confirm_frames=int(kick_confirm_frames),
    )
    diag.update(eval_diag)

    kick_time = eval_diag.get("selected_kick_time")
    kick_frame = eval_diag.get("selected_kick_frame")
    if not kick_time or kick_frame is None:
        return diag

    pre_kick = merged[merged["frame"] <= int(kick_frame)].copy()
    segments = find_setup_segments(
        pre_kick,
        goal_x_min,
        goal_x_max,
        goal_area_half_width_m,
        gk_ball_distance_m,
        min_setup_frames,
        setup_gap_frames,
    )
    if not segments:
        diag["setup_found"] = False
        diag["setup_start_time"] = str(kick_time)
        diag["setup_end_time"] = str(kick_time)
        diag["ready_time"] = str(kick_time)
    else:
        seg = segments[-1]
        diag["setup_found"] = True
        diag["setup_start_time"] = str(seg["time"].iloc[0])
        diag["setup_end_time"] = str(seg["time"].iloc[-1])

        ready_time = find_ready_time_in_setup(
            seg,
            ready_min_dist_m=ready_min_dist_m,
            ready_max_dist_m=ready_max_dist_m,
            stable_frames=ready_stable_frames,
            ball_step_eps_m=ready_ball_step_eps_m,
            gk_step_eps_m=ready_gk_step_eps_m,
        )
        diag["ready_time"] = str(ready_time or diag["setup_end_time"])
    diag["time_before_event_s"] = float(time_to_seconds(event_time_tracking) - time_to_seconds(str(kick_time)))
    diag["build_up_detected"] = True
    diag["failure_reason"] = "detected"
    return diag
