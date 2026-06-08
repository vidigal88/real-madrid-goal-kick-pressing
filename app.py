from __future__ import annotations
import streamlit as st
st.set_page_config(page_title="Real Madrid Pressing Analysis", layout="wide")

import json
import math
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import numpy as np



APP_ROOT = Path(__file__).resolve().parent

DEFAULT_PROCESSED_ROOT = "data/processed/rm_pressing"
DEFAULT_RAW_ROOT = "data/raw/RealMadrid"
DEFAULT_FEATURES_ROOT = "data/processed/rm_pressing_features"

DEFAULT_MIN_PLAYERS = 12
DEFAULT_INCLUDE_EXTRAPOLATED = True
DEFAULT_PRESSURE_THRESHOLD = 5.0


def _resolve_user_path(path_str: str) -> Path:
    """Resolve user-provided path string to absolute Path object.

    Converts relative paths to absolute paths relative to app root directory.
    Absolute paths are returned unchanged.

    Args:
        path_str: Path string (relative or absolute)

    Returns:
        Absolute Path object

    Example:
        >>> _resolve_user_path("data/processed")
        Path('/path/to/app/data/processed')
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (APP_ROOT / path).resolve()


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        ax: X coordinate of point A
        ay: Y coordinate of point A
        bx: X coordinate of point B
        by: Y coordinate of point B

    Returns:
        Distance in same units as input coordinates (typically meters)
    """
    return math.hypot(ax - bx, ay - by)


def time_to_seconds(tracking_time: str) -> float:
    """Convert tracking time format (HH:MM:SS.CC) to total seconds.

    SkillCorner tracking data uses format HH:MM:SS.CC where CC is centiseconds (hundredths).

    Args:
        tracking_time: Time string in format "HH:MM:SS.CC" (e.g., "00:05:23.45")

    Returns:
        Total seconds as float (e.g., 323.45)

    Example:
        >>> time_to_seconds("00:05:23.45")
        323.45
    """
    hh, mm, ss_dec = tracking_time.split(":")
    ss, dec = (ss_dec.split(".", 1) + ["0"])[:2]
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(dec) / 100.0


def seconds_to_time(seconds: float) -> str:
    """Convert total seconds to tracking time format (HH:MM:SS.CC).

    Inverse of time_to_seconds. Negative values are clamped to 00:00:00.00.

    Args:
        seconds: Total seconds (e.g., 323.45)

    Returns:
        Time string in format "HH:MM:SS.CC" (e.g., "00:05:23.45")

    Example:
        >>> seconds_to_time(323.45)
        '00:05:23.45'
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


@st.cache_data(show_spinner=False)
def load_processed_index(processed_root: str) -> pd.DataFrame:
    """Load the build-up index containing metadata for all extracted build-ups.

    The index provides essential metadata for each extracted build-up including:
    - build_up_id: Unique identifier
    - game_id: Match identifier
    - period: Match period (1, 2, etc.)
    - opponent_team_name: Opponent team name
    - ready_time, kick_time: Key event timestamps
    - window_start, window_end: Full build-up time window
    - frames_path: Relative path to tracking data parquet file

    Args:
        processed_root: Root directory containing processed data (index.parquet)

    Returns:
        DataFrame with one row per build-up and metadata columns

    Note:
        This function is cached to avoid reloading the index on every interaction.
        Cache is invalidated when processed_root changes.
    """
    path = _resolve_user_path(processed_root) / "index.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_processed_frames(processed_root: str, frames_rel_path: str) -> pd.DataFrame:
    """Load tracking data for a specific build-up window.

    Tracking data is stored in long format with columns:
    - time: Timestamp in tracking format (MM:SS.S)
    - frame: Frame number
    - player_id: Player identifier
    - team_id: Team identifier
    - x, y: Player position coordinates (meters, pitch center = origin)
    - is_detected: Boolean indicating if player was tracked in this frame

    Args:
        processed_root: Root directory containing processed data
        frames_rel_path: Relative path to build-up parquet file (e.g., "frames/build_up_00123.parquet")

    Returns:
        DataFrame with tracking data in long format (one row per player per frame)

    Note:
        Cached to avoid reloading the same build-up when user changes visualization settings.
    """
    path = _resolve_user_path(processed_root) / frames_rel_path
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_meta(game_id: str, raw_root: str) -> dict[str, Any]:
    """Load match metadata from SkillCorner JSON format.

    Metadata includes:
    - home_team, away_team: Team information (id, name, acronym)
    - players: List of player dictionaries (id, team_id, short_name, player_role, etc.)
    - pitch_length, pitch_width: Pitch dimensions in meters
    - periods: Period information

    Args:
        game_id: Match identifier (e.g., "2014987")
        raw_root: Root directory containing raw SkillCorner data

    Returns:
        Dictionary with match metadata

    Note:
        Cached per game_id to avoid repeated file reads. Meta files are typically 100-500 KB.
    """
    path = _resolve_user_path(raw_root) / "meta" / f"{game_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_features(features_root: str) -> pd.DataFrame:
    """Load computed feature matrix for all build-ups.

    Features include 20+ numerical metrics per build-up:
    - Pressure metrics (7): t_first_pressure_s, pressure_frames_ratio, etc.
    - Compactness metrics (6): rm_width_mean_m, rm_hull_area_mean_m2, etc.
    - Goal kick features (3): goal_kick_type, gk_kick_distance_m, receiver_lane
    - Steering features (2), Outcome features (2), QC features (3)

    Args:
        features_root: Root directory containing features.parquet file

    Returns:
        DataFrame with one row per build-up and feature columns.
        Returns empty DataFrame if features.parquet doesn't exist.

    Note:
        Features are optional - app works without them but dashboard and metrics
        panels will show limited information. Generate features using main.py.
    """
    path = _resolve_user_path(features_root) / "features.parquet"
    print(f"[DEBUG] load_features: path={path}, exists={path.exists()}")
    if not path.exists():
        print("[DEBUG] Features file does NOT exist!")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"[DEBUG] Loaded features: shape={df.shape}, build_up_ids={sorted(df['build_up_id'].unique().tolist()[:10])}")
    if 't_first_pressure_s' in df.columns:
        print(f"[DEBUG] Sample pressure data (first 10): {df[['build_up_id', 't_first_pressure_s']].head(10).to_dict('records')}")
    return df


def get_build_up_features(features_df: pd.DataFrame, build_up_id: int) -> dict[str, Any]:
    """Extract features for a specific build-up as a dictionary.

    Args:
        features_df: Full feature matrix DataFrame (from load_features)
        build_up_id: Build-up identifier to extract features for

    Returns:
        Dictionary mapping feature names to values for the specified build-up.
        Returns empty dict if build-up not found or features_df is empty.

    Example:
        >>> features_df = load_features("data/processed/rm_pressing_features")
        >>> features = get_build_up_features(features_df, 123)
        >>> print(features['t_first_pressure_s'])
        2.4
    """
    print(f"[DEBUG] get_build_up_features called: build_up_id={build_up_id}, features_df.shape={features_df.shape if not features_df.empty else 'EMPTY'}")
    if features_df.empty:
        print("[DEBUG] Features DataFrame is EMPTY!")
        return {}
    row = features_df[features_df["build_up_id"] == build_up_id]
    if row.empty:
        print(f"[DEBUG] No features found for build_up_id={build_up_id}")
        print(f"[DEBUG] Available build_up_ids: {sorted(features_df['build_up_id'].unique().tolist()[:10])}")
        return {}
    features_dict = row.iloc[0].to_dict()
    print(f"[DEBUG] Found features: t_first_pressure_s={features_dict.get('t_first_pressure_s')}, pressure_frames_ratio={features_dict.get('pressure_frames_ratio')}")
    return features_dict


def build_player_mapping(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for p in meta.get("players", []):
        name = p.get("short_name")
        if not name:
            name = f"{(p.get('first_name') or '').strip()} {(p.get('last_name') or '').strip()}".strip() or "Player"

        mapping[int(p["id"])] = {
            "name": name,
            "number": p.get("number"),
            "team_id": p.get("team_id"),
            "role": (p.get("player_role") or {}).get("acronym"),
        }
    return mapping


def estimate_fps_from_frames(frames: list[dict[str, Any]], default_fps: int = 10) -> int:
    if len(frames) < 3:
        return int(default_fps)

    ts = [time_to_seconds(str(fr.get("timestamp"))) for fr in frames]
    deltas = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return int(default_fps)

    deltas.sort()
    median = deltas[len(deltas) // 2]
    if median <= 0:
        return int(default_fps)

    fps = int(round(1.0 / median))
    return int(max(1, min(60, fps)))


def build_window_frames_from_long_df(
    *,
    df: pd.DataFrame,
    period: int,
    min_players: int,
    include_extrapolated: bool,
    require_ball_detected: bool,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    df = df[df["period"] == int(period)][["time", "frame", "player_id", "is_ball", "is_detected", "x", "y"]].copy()
    if df.empty:
        return []

    frames_out: list[dict[str, Any]] = []
    roster = sorted({int(pid) for pid in df.loc[df["is_ball"] == False, "player_id"].unique()})

    # Debug: Show roster information
    print(f"Frame building: Found {len(roster)} unique players in tracking data for period {period}")
    print(f"  Roster player IDs: {roster}")

    last_pos: dict[int, tuple[float, float]] = {}
    last_ball: tuple[float, float] | None = None

    for (time_str, frame_num), g in df.groupby(["time", "frame"], sort=True):
        ball_data = {"is_detected": False, "x": None, "y": None}
        players: dict[int, dict[str, Any]] = {}

        for r in g.itertuples(index=False):
            if bool(r.is_ball):
                if r.x is None or r.y is None:
                    continue
                if bool(r.is_detected) or bool(include_extrapolated):
                    ball_data = {"is_detected": bool(r.is_detected), "x": float(r.x), "y": float(r.y)}
                    last_ball = (float(r.x), float(r.y))
                continue

            pid = int(r.player_id)
            if r.x is None or r.y is None:
                continue

            if not bool(include_extrapolated) and not bool(r.is_detected):
                continue

            players[pid] = {"player_id": pid, "is_detected": bool(r.is_detected), "x": float(r.x), "y": float(r.y)}
            last_pos[pid] = (float(r.x), float(r.y))

        if bool(include_extrapolated):
            for pid in roster:
                if pid in players:
                    continue
                if pid in last_pos:
                    x, y = last_pos[pid]
                    players[pid] = {"player_id": pid, "is_detected": False, "x": float(x), "y": float(y)}
            if last_ball is not None and ball_data["x"] is None and ball_data["y"] is None:
                bx, by = last_ball
                ball_data = {"is_detected": False, "x": float(bx), "y": float(by)}

        if bool(require_ball_detected) and not bool(ball_data.get("is_detected")):
            continue

        detected_players = sum(1 for p in players.values() if bool(p.get("is_detected")))
        players_for_filter = len(players) if bool(include_extrapolated) else detected_players
        if int(min_players) > 0 and players_for_filter < int(min_players):
            continue

        frames_out.append(
            {
                "timestamp": str(time_str),
                "period": int(period),
                "frame": int(frame_num),
                "player_data": list(players.values()),
                "ball_data": ball_data,
            }
        )

    # Debug: Show summary statistics
    if frames_out:
        player_counts = [len(f["player_data"]) for f in frames_out]
        detected_counts = [sum(1 for p in f["player_data"] if p["is_detected"]) for f in frames_out]
        print(f"Frame building: Generated {len(frames_out)} frames")
        print(f"  Player count per frame: min={min(player_counts)}, max={max(player_counts)}, avg={sum(player_counts)/len(player_counts):.1f}")
        print(f"  Detected players per frame: min={min(detected_counts)}, max={max(detected_counts)}, avg={sum(detected_counts)/len(detected_counts):.1f}")
        print(f"  Extrapolation enabled: {include_extrapolated}")

    return frames_out


def fill_frame_gaps_linear(*, frames: list[dict[str, Any]], fps: int, max_gap_seconds: float) -> list[dict[str, Any]]:
    if len(frames) < 2:
        return frames

    fps = int(max(1, fps))
    max_gap_seconds = float(max_gap_seconds)
    step = 1.0 / float(fps)

    def _to_pos_map(fr: dict[str, Any]) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for p in fr.get("player_data", []):
            out[int(p["player_id"])] = (float(p["x"]), float(p["y"]))
        return out

    def _ball_pos(fr: dict[str, Any]) -> tuple[float, float] | None:
        b = fr.get("ball_data") or {}
        if b.get("x") is None or b.get("y") is None:
            return None
        return float(b["x"]), float(b["y"])

    out: list[dict[str, Any]] = []
    for a, b in zip(frames, frames[1:]):
        out.append(a)
        a_sec = time_to_seconds(str(a["timestamp"]))
        b_sec = time_to_seconds(str(b["timestamp"]))
        gap = b_sec - a_sec

        if gap <= step or gap > max_gap_seconds:
            continue

        n = int(round(gap / step)) - 1
        if n <= 0:
            continue

        a_map = _to_pos_map(a)
        b_map = _to_pos_map(b)
        ids = sorted(set(a_map) | set(b_map))

        a_ball = _ball_pos(a)
        b_ball = _ball_pos(b)

        for k in range(1, n + 1):
            ratio = (k * step) / gap
            ts = seconds_to_time(a_sec + k * step)
            player_data = []
            for pid in ids:
                ax, ay = a_map.get(pid, b_map.get(pid, (None, None)))
                bx, by = b_map.get(pid, a_map.get(pid, (None, None)))
                if ax is None or ay is None or bx is None or by is None:
                    continue
                x = float(ax) + (float(bx) - float(ax)) * ratio
                y = float(ay) + (float(by) - float(ay)) * ratio
                player_data.append({"player_id": int(pid), "is_detected": False, "x": float(x), "y": float(y)})

            ball_data = {"is_detected": False, "x": None, "y": None}
            if a_ball is not None and b_ball is not None:
                ax, ay = a_ball
                bx, by = b_ball
                ball_data = {
                    "is_detected": False,
                    "x": float(ax) + (float(bx) - float(ax)) * ratio,
                    "y": float(ay) + (float(by) - float(ay)) * ratio,
                }

            out.append(
                {
                    "timestamp": ts,
                    "period": int(a.get("period")),
                    "frame": int(a.get("frame", 0)) + k,
                    "player_data": player_data,
                    "ball_data": ball_data,
                }
            )

    out.append(frames[-1])
    return out


def create_pitch_shapes(pitch_length: float, pitch_width: float) -> list[dict[str, Any]]:
    hl = pitch_length / 2.0
    hw = pitch_width / 2.0

    penalty_depth = 16.5
    penalty_width = 40.3
    goal_area_depth = 5.5
    goal_area_width = 18.3

    shapes: list[dict[str, Any]] = []

    def line(x0, y0, x1, y1, width=2):
        shapes.append({"type": "line", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": {"color": "white", "width": width}})

    def rect(x0, y0, x1, y1, width=2):
        shapes.append(
            {"type": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": {"color": "white", "width": width}, "fillcolor": "rgba(0,0,0,0)"}
        )

    rect(-hl, -hw, hl, hw, width=2)
    line(0, -hw, 0, hw, width=2)
    shapes.append({"type": "circle", "x0": -9.15, "y0": -9.15, "x1": 9.15, "y1": 9.15, "line": {"color": "white", "width": 2}})
    rect(-hl, -penalty_width / 2, -hl + penalty_depth, penalty_width / 2, width=2)
    rect(hl - penalty_depth, -penalty_width / 2, hl, penalty_width / 2, width=2)
    rect(-hl, -goal_area_width / 2, -hl + goal_area_depth, goal_area_width / 2, width=2)
    rect(hl - goal_area_depth, -goal_area_width / 2, hl, goal_area_width / 2, width=2)

    return shapes


def create_animation_figure(
    *,
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    player_map: dict[int, dict[str, Any]],
    rm_team_id: int,
    opponent_team_id: int,
    title_line: str,
    kick_time: str,
    fps: int,
    show_rm_rings: bool,
    rm_ring_size: int,
    rm_ring_width: int,
    rm_ring_color: str = "red",
    show_pressing_network: bool = True,
    show_pressure_zones: bool = True,
    pressure_threshold_m: float = 5.0,
    trigger_info: Optional[dict[str, Any]] = None,
    show_convex_hull: bool = False,
    show_opp_hull: bool = False,
    show_voronoi: bool = False,
    show_vectors: bool = False,
) -> go.Figure | None:
    if not frames:
        return None

    # Lazy import to avoid Streamlit execution at module import time
    from src.viz.pressing_network import (
        calculate_pressing_links,
        create_network_traces,
        create_pressure_zone_shapes,
        create_pressure_zone_traces,
        get_ball_carrier_for_frame,
    )

    pitch_length = float(meta.get("pitch_length") or 105.0)
    pitch_width = float(meta.get("pitch_width") or 68.0)
    hl = pitch_length / 2.0
    hw = pitch_width / 2.0

    def split_frame(fr: dict[str, Any], frame_idx: int):
        rm_x, rm_y, rm_numbers, rm_names, rm_ids = [], [], [], [], []
        opp_x, opp_y, opp_numbers, opp_names = [], [], [], []

        for p in fr.get("player_data", []):
            pid = int(p["player_id"])
            info = player_map.get(pid)
            if not info:
                continue
            team_id = int(info.get("team_id") or -999)
            if team_id == int(rm_team_id):
                rm_x.append(float(p["x"]))
                rm_y.append(float(p["y"]))
                rm_numbers.append(str(info.get("number") or ""))
                rm_names.append(str(info.get("name") or ""))
                rm_ids.append(pid)
            elif team_id == int(opponent_team_id):
                opp_x.append(float(p["x"]))
                opp_y.append(float(p["y"]))
                opp_numbers.append(str(info.get("number") or ""))
                opp_names.append(str(info.get("name") or ""))

        ball = fr.get("ball_data") or {}
        bx = float(ball["x"]) if ball.get("x") is not None else None
        by = float(ball["y"]) if ball.get("y") is not None else None
        
        # Identify ball carrier and pressing players
        carrier_id = get_ball_carrier_for_frame(fr, player_map, rm_team_id, opponent_team_id)
        pressing = calculate_pressing_links(fr, player_map, rm_team_id, carrier_id, pressure_threshold_m) if carrier_id else []

        # Debug: Show carrier and pressing detection (only for first frame)
        if frame_idx == 0:
            print(f"Carrier detection: carrier_id={carrier_id}, pressing_players={len(pressing)}")
            if carrier_id:
                carrier_info = player_map.get(carrier_id, {})
                print(f"  Carrier: #{carrier_info.get('number')} {carrier_info.get('name')}")
            if pressing:
                pressing_numbers = [player_map.get(p['player_id'], {}).get('number', '?') for p in pressing[:3]]
                print(f"  Pressing players: {', '.join(f'#{n}' for n in pressing_numbers)}")
            else:
                print(f"  No pressing players within {pressure_threshold_m}m threshold")

        # Find carrier position for network lines
        carrier_pos = None
        if carrier_id:
            for p in fr.get("player_data", []):
                if int(p["player_id"]) == carrier_id:
                    carrier_pos = (float(p["x"]), float(p["y"]))
                    break
        
        return (rm_x, rm_y, rm_numbers, rm_names, rm_ids), (opp_x, opp_y, opp_numbers, opp_names), (bx, by), carrier_id, carrier_pos, pressing

    first = frames[0]
    (rm_x, rm_y, rm_numbers, rm_names, rm_ids), (opp_x, opp_y, opp_numbers, opp_names), (bx, by), carrier_id, carrier_pos, pressing = split_frame(first, 0)

    fig = go.Figure()

    # Set pitch shapes (static layout - no pressure zones here, they're animated)
    fig.update_layout(shapes=create_pitch_shapes(pitch_length, pitch_width))

    # Add RM rings
    if bool(show_rm_rings):
        fig.add_trace(
            go.Scatter(
                x=rm_x,
                y=rm_y,
                mode="markers",
                name="Real Madrid (ring)",
                marker={"size": int(rm_ring_size), "color": "rgba(0,0,0,0)", "line": {"color": rm_ring_color, "width": int(rm_ring_width)}},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Add RM players with trigger highlighting
    marker_colors = []
    marker_sizes = []
    for i, pid in enumerate(rm_ids):
        is_trigger = trigger_info and int(trigger_info.get("player_id", -1)) == pid
        marker_colors.append("yellow" if is_trigger else "white")
        marker_sizes.append(16 if is_trigger else 12)
    
    fig.add_trace(
        go.Scatter(
            x=rm_x,
            y=rm_y,
            mode="markers+text",
            name="Real Madrid",
            text=rm_numbers,
            textposition="middle center",
            textfont={"color": "black", "size": 11, "family": "Arial Black"},
            marker={"size": marker_sizes, "color": marker_colors, "line": {"color": "black", "width": 1}},
            hovertext=rm_names,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Add opponent players
    fig.add_trace(
        go.Scatter(
            x=opp_x,
            y=opp_y,
            mode="markers+text",
            name="Opponent",
            text=opp_numbers,
            textposition="middle center",
            textfont={"color": "white", "size": 11},
            marker={"size": 12, "color": "#1f77b4", "line": {"color": "white", "width": 1}},
            hovertext=opp_names,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Add ball
    fig.add_trace(
        go.Scatter(
            x=[bx] if bx is not None else [],
            y=[by] if by is not None else [],
            mode="markers",
            name="Ball",
            marker={"size": 8, "color": "orange", "line": {"color": "black", "width": 1}},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Add initial overlays (pressure zones, pressing network, convex hull, voronoi, vectors)
    # These need to be in the initial figure for Plotly animation to work correctly

    # Add pressure zones (animated per frame)
    if show_pressure_zones:
        zone_traces = create_pressure_zone_traces(carrier_pos, pressure_threshold_m, show_pressure_zones)
        for trace in zone_traces:
            fig.add_trace(trace)

    # Add pressing network
    if show_pressing_network and carrier_pos and pressing:
        network_traces = create_network_traces(pressing, carrier_pos, show_pressing_network)
        for trace in network_traces:
            fig.add_trace(trace)

    if show_convex_hull:
        hull_points = compute_convex_hull_overlay(first, rm_team_id, player_map)
        if hull_points:
            hull_x = [p[0] for p in hull_points]
            hull_y = [p[1] for p in hull_points]
            fig.add_trace(go.Scatter(
                x=hull_x,
                y=hull_y,
                mode="lines",
                name="RM Convex Hull",
                line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.1)",
                hoverinfo="skip",
                showlegend=False,
            ))

    if show_opp_hull:
        opp_hull_points = compute_convex_hull_overlay(first, opponent_team_id, player_map)
        if opp_hull_points:
            opp_hull_x = [p[0] for p in opp_hull_points]
            opp_hull_y = [p[1] for p in opp_hull_points]
            fig.add_trace(go.Scatter(
                x=opp_hull_x,
                y=opp_hull_y,
                mode="lines",
                name="Opp Convex Hull",
                line=dict(color="rgba(0, 80, 200, 0.8)", width=2, dash="dash"),
                fill="toself",
                fillcolor="rgba(0, 80, 200, 0.2)",
                hoverinfo="skip",
                showlegend=False,
            ))

    if show_voronoi:
        voronoi_data = compute_voronoi_overlay(first, player_map, pitch_length, pitch_width, rm_team_id)
        if voronoi_data:
            voronoi_traces = create_voronoi_traces(voronoi_data)
            for trace in voronoi_traces:
                fig.add_trace(trace)

    if show_vectors:
        vectors = compute_movement_vectors(frames, 0, player_map, lookback=1)
        print(f"Rendering vectors: {len(vectors)} vectors to render in initial figure")
        # Scale factor to make vectors more visible (10x amplification)
        vector_scale = 10.0
        for vx, vy, dx, dy, team_id in vectors:
            # Scale up the vector for visibility
            dx_scaled = dx * vector_scale
            dy_scaled = dy * vector_scale

            color = "rgba(255, 0, 0, 0.8)" if team_id == rm_team_id else "rgba(0, 100, 255, 0.8)"
            fig.add_trace(go.Scatter(
                x=[vx, vx + dx_scaled],
                y=[vy, vy + dy_scaled],
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[vx + dx_scaled],
                y=[vy + dy_scaled],
                mode="markers",
                marker=dict(
                    symbol="arrow",
                    size=20,
                    color=color,
                    angle=math.degrees(math.atan2(dy_scaled, dx_scaled)),
                    angleref="previous",
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Build frames with pressing network
    plot_frames: list[go.Frame] = []
    for idx, fr in enumerate(frames):
        (rm_x, rm_y, rm_numbers, rm_names, rm_ids), (opp_x, opp_y, opp_numbers, opp_names), (bx, by), carrier_id, carrier_pos, pressing = split_frame(fr, idx)
        
        data = []
        
        # RM rings
        if bool(show_rm_rings):
            data.append(go.Scatter(x=rm_x, y=rm_y))
        
        # RM players with trigger highlighting
        marker_colors = []
        marker_sizes = []
        for i, pid in enumerate(rm_ids):
            is_trigger = trigger_info and int(trigger_info.get("player_id", -1)) == pid
            marker_colors.append("yellow" if is_trigger else "white")
            marker_sizes.append(16 if is_trigger else 12)
        
        data.append(go.Scatter(
            x=rm_x, 
            y=rm_y, 
            text=rm_numbers, 
            hovertext=rm_names,
            marker={"size": marker_sizes, "color": marker_colors, "line": {"color": "black", "width": 1}},
        ))
        
        # Opponent players
        data.append(go.Scatter(x=opp_x, y=opp_y, text=opp_numbers, hovertext=opp_names))
        
        # Ball
        data.append(go.Scatter(x=[bx] if bx is not None else [], y=[by] if by is not None else []))

        # Add pressure zones (animated per frame)
        if show_pressure_zones:
            zone_traces = create_pressure_zone_traces(carrier_pos, pressure_threshold_m, show_pressure_zones)
            data.extend(zone_traces)

        # Add pressing network lines
        if show_pressing_network and carrier_pos and pressing:
            network_traces = create_network_traces(pressing, carrier_pos, show_pressing_network)
            data.extend(network_traces)

        # Add convex hull overlays
        if show_convex_hull:
            hull_points = compute_convex_hull_overlay(fr, rm_team_id, player_map)
            if hull_points:
                hull_x = [p[0] for p in hull_points]
                hull_y = [p[1] for p in hull_points]
                data.append(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode="lines",
                    name="RM Convex Hull",
                    line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    hoverinfo="skip",
                    showlegend=False,
                ))

        if show_opp_hull:
            opp_hull_points = compute_convex_hull_overlay(fr, opponent_team_id, player_map)
            if opp_hull_points:
                opp_hull_x = [p[0] for p in opp_hull_points]
                opp_hull_y = [p[1] for p in opp_hull_points]
                data.append(go.Scatter(
                    x=opp_hull_x,
                    y=opp_hull_y,
                    mode="lines",
                    name="Opp Convex Hull",
                    line=dict(color="rgba(0, 80, 200, 0.8)", width=2, dash="dash"),
                    fill="toself",
                    fillcolor="rgba(0, 80, 200, 0.2)",
                    hoverinfo="skip",
                    showlegend=False,
                ))

        # Add Voronoi diagram
        if show_voronoi:
            voronoi_data = compute_voronoi_overlay(fr, player_map, pitch_length, pitch_width, rm_team_id)
            if voronoi_data:
                voronoi_traces = create_voronoi_traces(voronoi_data)
                data.extend(voronoi_traces)

        # Add movement vectors
        if show_vectors:
            vectors = compute_movement_vectors(frames, idx, player_map, lookback=1)
            # Scale factor to make vectors more visible (10x amplification)
            vector_scale = 10.0
            for vx, vy, dx, dy, team_id in vectors:
                # Scale up the vector for visibility
                dx_scaled = dx * vector_scale
                dy_scaled = dy * vector_scale

                color = "rgba(255, 0, 0, 0.8)" if team_id == rm_team_id else "rgba(0, 100, 255, 0.8)"
                data.append(go.Scatter(
                    x=[vx, vx + dx_scaled],
                    y=[vy, vy + dy_scaled],
                    mode="lines",
                    line=dict(color=color, width=4),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                # Add arrowhead
                data.append(go.Scatter(
                    x=[vx + dx_scaled],
                    y=[vy + dy_scaled],
                    mode="markers",
                    marker=dict(
                        symbol="arrow",
                        size=20,
                        color=color,
                        angle=math.degrees(math.atan2(dy_scaled, dx_scaled)),
                        angleref="previous",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ))

        plot_frames.append(go.Frame(data=data, name=str(fr.get("timestamp"))))

    fig.frames = plot_frames

    title = f"{title_line}<br><sup>kick: {kick_time} | frames: {len(frames)} | fps: {int(fps)}</sup>"

    fig.update_layout(
        title=title,
        xaxis={"range": [-hl, hl], "showgrid": False, "zeroline": False, "visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"range": [-hw, hw], "showgrid": False, "zeroline": False, "visible": False},
        plot_bgcolor="#2a7f38",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "x": 0.02,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": int(1000 / max(1, int(fps))), "redraw": True}, "fromcurrent": True}],
                    },
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.15,
                "y": 0.02,
                "len": 0.83,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"t": 10, "b": 0},
                "currentvalue": {"prefix": "t="},
                "steps": [{"args": [[fr.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": fr.name, "method": "animate"} for fr in plot_frames],
            }
        ],
    )

    return fig




def render_info_panel(
    *,
    features: dict[str, Any],
    trigger_info: Optional[dict[str, Any]],
    player_map: dict[int, dict[str, Any]],
) -> None:
    """Render build-up analysis information panel in the viewer.

    Displays key metrics and characteristics for the selected build-up in a 3-column layout:
    - Column 1: Build-up type (short/long), kick distance, receiver lane
    - Column 2: Trigger player information (first presser and timing)
    - Column 3: Pressing metrics (pressure intensity, team compactness)

    Args:
        features: Dictionary of computed features for the build-up (from get_build_up_features)
        trigger_info: Optional trigger player information dictionary containing:
            - player_id: Player identifier
            - name: Player name
            - number: Jersey number
            - timestamp: Time of first pressure
            - distance: Distance to ball carrier at trigger moment
        player_map: Mapping of player_id to player metadata (team_id, name, number)

    Note:
        If features dict is empty, only trigger information will be displayed.
        If trigger_info is None, panel shows "No trigger detected".
    """
    st.subheader("Build-Up Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Build-up type
        gk_type = features.get("goal_kick_type", "unknown")
        kick_dist = features.get('gk_kick_distance_m', 0)

        if gk_type == "long":
            st.markdown(f"### Long Kick ({kick_dist:.1f}m)")
        elif gk_type == "short":
            st.markdown(f"### Short Kick ({kick_dist:.1f}m)")
        else:
            st.markdown("### Unknown Type")

        lane = features.get("receiver_lane", "").title()
        if lane and lane != "Unknown":
            st.markdown(f"**Lane:** {lane}")

        time_to_recv = features.get("time_to_receiver_s")
        if time_to_recv and not np.isnan(time_to_recv):
            st.markdown(f"**Reception:** {time_to_recv:.1f}s")

    with col2:
        # Trigger player
        st.markdown("### Trigger Player")
        if trigger_info:
            player_name = trigger_info.get("name", "Unknown")
            player_num = trigger_info.get("number", "?")
            trigger_time = trigger_info.get("timestamp", "N/A")
            trigger_dist = trigger_info.get("distance", 0)

            st.markdown(f"**#{player_num} {player_name}**")
            st.markdown(f"{trigger_time} ({trigger_dist:.1f}m)")
        else:
            st.markdown("*No trigger*")

    with col3:
        # Pressure metrics
        st.markdown("### Pressure Metrics")

        t_first = features.get("t_first_pressure_s")
        t_first_known = t_first is not None and not pd.isna(t_first)
        if t_first_known:
            st.markdown(f"**First:** {float(t_first):.1f}s")
        else:
            st.markdown("**First:** N/A")

        pressure_ratio = features.get("pressure_frames_ratio")
        if pressure_ratio is None or pd.isna(pressure_ratio):
            pressure_ratio = 0.0
        st.markdown(f"**Intensity:** {float(pressure_ratio) * 100:.0f}%")

        bursts = features.get("pressure_bursts_n")
        if bursts is None or pd.isna(bursts):
            bursts = 0
        st.markdown(f"**Bursts:** {int(float(bursts))}")


def _render_frame_inspector(
    *,
    frames: list[dict[str, Any]],
    player_map: dict[int, dict[str, Any]],
    rm_team_id: int,
    opponent_team_id: int,
    gk_id: int,
    df_long: pd.DataFrame,
) -> None:
    if not frames:
        return

    with st.expander("Frame inspector (debug)", expanded=False):
        default_idx = 18 if len(frames) > 18 else 0
        i = st.slider("Frame index", 0, len(frames) - 1, default_idx)
        fr = frames[int(i)]

        ball = fr.get("ball_data") or {}
        ball_x = float(ball["x"]) if ball.get("x") is not None else None
        ball_y = float(ball["y"]) if ball.get("y") is not None else None
        ball_detected = bool(ball.get("is_detected"))

        gk_x = gk_y = None
        for p in fr.get("player_data", []):
            if int(p.get("player_id")) == int(gk_id):
                gk_x = float(p["x"])
                gk_y = float(p["y"])
                break

        dist = None
        if ball_x is not None and ball_y is not None and gk_x is not None and gk_y is not None:
            dist = _distance(ball_x, ball_y, gk_x, gk_y)

        detected_players = sum(1 for p in fr.get("player_data", []) if bool(p.get("is_detected")))
        st.json(
            {
                "timestamp": fr.get("timestamp"),
                "period": fr.get("period"),
                "frame": fr.get("frame"),
                "ball": {"detected": ball_detected, "x": ball_x, "y": ball_y},
                "gk": {"player_id": int(gk_id), "x": gk_x, "y": gk_y},
                "ball_gk_distance_m": dist,
                "n_players_total": len(fr.get("player_data", [])),
                "n_players_detected": int(detected_players),
            }
        )

        rows: list[dict[str, Any]] = []
        for p in fr.get("player_data", []):
            pid = int(p["player_id"])
            info = player_map.get(pid, {})
            team_id = info.get("team_id")
            team = "Other"
            if team_id == rm_team_id:
                team = "Real Madrid"
            elif team_id == opponent_team_id:
                team = "Opponent"

            d_ball = None
            if ball_x is not None and ball_y is not None:
                d_ball = _distance(float(p["x"]), float(p["y"]), float(ball_x), float(ball_y))

            rows.append(
                {
                    "team": team,
                    "player_id": pid,
                    "number": info.get("number"),
                    "name": info.get("name"),
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "is_detected": bool(p.get("is_detected")),
                    "dist_to_ball": d_ball,
                }
            )

        df_players = pd.DataFrame(rows)
        if not df_players.empty:
            df_players = df_players.sort_values(["team", "dist_to_ball"], na_position="last", ignore_index=True)
        st.dataframe(df_players, use_container_width=True, hide_index=True)

        t = str(fr.get("timestamp"))
        fnum = int(fr.get("frame"))
        raw = df_long[(df_long["time"] == t) & (df_long["frame"] == fnum)].copy()
        if not raw.empty:
            st.caption("Frame tracking data")
            st.dataframe(raw, use_container_width=True, hide_index=True)


def render_pressure_timeline(
    frames: list[dict[str, Any]],
    rm_team_id: int,
    opponent_team_id: int,
    player_map: dict[int, dict[str, Any]],
    pressure_threshold_m: float = 5.0,
) -> go.Figure:
    """Render pressure intensity timeline chart.

    Computes pressure intensity for each frame by counting how many opponent
    players are within the pressure threshold of any Real Madrid player.

    Args:
        frames: List of frame dictionaries with tracking data
        rm_team_id: Real Madrid team ID
        opponent_team_id: Opponent team ID
        player_map: Player ID to metadata mapping
        pressure_threshold_m: Distance threshold for pressure (meters)

    Returns:
        Plotly Figure with timeline chart
    """
    if not frames:
        return go.Figure()

    timestamps = []
    pressure_counts = []

    for fr in frames:
        timestamp = fr.get("timestamp", "")
        timestamps.append(timestamp)

        # Get RM and opponent positions
        rm_positions = []
        opp_positions = []

        for p in fr.get("player_data", []):
            pid = int(p["player_id"])
            info = player_map.get(pid)
            if not info:
                continue
            team_id = int(info.get("team_id") or -999)
            if not bool(p.get("is_detected")):
                continue
            x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue

            if team_id == rm_team_id:
                rm_positions.append((float(x), float(y)))
            elif team_id == opponent_team_id:
                opp_positions.append((float(x), float(y)))

        # Count opponents under pressure
        under_pressure = 0
        for opp_x, opp_y in opp_positions:
            for rm_x, rm_y in rm_positions:
                dist = math.hypot(opp_x - rm_x, opp_y - rm_y)
                if dist <= pressure_threshold_m:
                    under_pressure += 1
                    break  # Count each opponent only once

        pressure_counts.append(under_pressure)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(timestamps))),
            y=pressure_counts,
            mode="lines+markers",
            name="Players Under Pressure",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            hovertemplate="Frame: %{x}<br>Under Pressure: %{y}<br>Time: %{text}<extra></extra>",
            text=timestamps,
        )
    )

    fig.update_layout(
        title="Pressure Intensity Over Time",
        xaxis_title="Frame",
        yaxis_title="Opponent Players Under Pressure",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )

    return fig


def compute_pressure_metrics_from_frames(
    *,
    frames: list[dict[str, Any]],
    player_map: dict[int, dict[str, Any]],
    rm_team_id: int,
    opponent_team_id: int,
    pressure_threshold_m: float,
    burst_gap_frames: int = 2,
) -> dict[str, Any]:
    """Compute pressure metrics directly from the current frames selection.

    This mirrors the ball-carrier + pressing-player logic used in the animation overlay
    (via src.viz.pressing_network) so the sidebar "Pressure Metrics" panel updates
    with the selected build-up window and the current pressure threshold.

    Returns keys compatible with the feature pipeline:
    - t_first_pressure_s (float, NaN if none)
    - pressure_frames_ratio (float 0-1)
    - pressure_bursts_n (int)
    """
    if not frames:
        return {"t_first_pressure_s": float("nan"), "pressure_frames_ratio": 0.0, "pressure_bursts_n": 0}

    from src.viz.pressing_network import calculate_pressing_links, get_ball_carrier_for_frame

    under_pressure: list[bool] = []
    times_s: list[float] = []

    for fr in frames:
        carrier_id = get_ball_carrier_for_frame(fr, player_map, rm_team_id, opponent_team_id)
        if not carrier_id:
            continue

        pressing = calculate_pressing_links(fr, player_map, rm_team_id, carrier_id, float(pressure_threshold_m))
        under_pressure.append(bool(pressing))
        try:
            times_s.append(time_to_seconds(str(fr.get("timestamp"))))
        except Exception:
            # Fallback: if timestamp format is unexpected, approximate by index later
            times_s.append(float("nan"))

    if not under_pressure:
        return {"t_first_pressure_s": float("nan"), "pressure_frames_ratio": 0.0, "pressure_bursts_n": 0}

    # Intensity: proportion of carrier-frames under pressure
    pressure_ratio = float(sum(1 for v in under_pressure if v)) / float(len(under_pressure))

    # Time to first pressure (relative to first carrier-frame)
    t_first = float("nan")
    if any(under_pressure):
        first_idx = next(i for i, v in enumerate(under_pressure) if v)
        t0 = times_s[0]
        t1 = times_s[first_idx]
        if not (pd.isna(t0) or pd.isna(t1)):
            t_first = float(max(0.0, t1 - t0))
        else:
            # Timestamp missing: fall back to 0.1s per frame (SkillCorner default ~10 FPS)
            t_first = float(first_idx) * 0.1

    # Bursts: count distinct episodes allowing short gaps
    true_idxs = [i for i, v in enumerate(under_pressure) if v]
    bursts = 0
    if true_idxs:
        bursts = 1
        max_gap = int(max(0, burst_gap_frames)) + 1
        for a, b in zip(true_idxs, true_idxs[1:]):
            if (b - a) > max_gap:
                bursts += 1

    return {
        "t_first_pressure_s": t_first,
        "pressure_frames_ratio": float(pressure_ratio),
        "pressure_bursts_n": int(bursts),
    }


def compute_convex_hull_overlay(
    frame: dict[str, Any],
    team_id: int,
    player_map: dict[int, dict[str, Any]],
) -> Optional[list[tuple[float, float]]]:
    """Compute convex hull for a team's positions in a single frame.

    Args:
        frame: Frame dictionary with tracking data
        team_id: Team ID to compute hull for
        player_map: Player ID to metadata mapping

    Returns:
        List of (x, y) coordinates forming the convex hull polygon,
        or None if insufficient players (<3)

    Note:
        A convex hull only includes the OUTERMOST boundary points.
        If you have 10 players but only 5 are on the perimeter,
        the hull will only show those 5 vertices. This is correct behavior.
    """
    from scipy.spatial import ConvexHull, QhullError

    positions = []
    player_ids = []  # Track which players we're considering

    for p in frame.get("player_data", []):
        pid = int(p["player_id"])
        info = player_map.get(pid)
        if not info:
            continue
        p_team_id = int(info.get("team_id") or -999)
        # Include ALL players (detected and extrapolated) for complete hull
        if p_team_id == team_id:
            x, y = p.get("x"), p.get("y")
            if x is not None and y is not None:
                positions.append((float(x), float(y)))
                player_ids.append(pid)

    if len(positions) < 3:
        return None

    try:
        points = np.array(positions)
        hull = ConvexHull(points)

        # Debug: Print stats (only in development)
        # Uncomment to see how many players vs hull vertices:
        # print(f"Total players considered: {len(positions)}, Hull vertices: {len(hull.vertices)}")
        # print(f"Players on hull: {[player_ids[i] for i in hull.vertices]}")

        # Return hull vertices in order
        hull_points = [(points[i, 0], points[i, 1]) for i in hull.vertices]
        # Close the polygon
        hull_points.append(hull_points[0])
        return hull_points
    except (QhullError, ValueError):
        return None


def compute_voronoi_overlay(
    frame: dict[str, Any],
    player_map: dict[int, dict[str, Any]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    rm_team_id: int = -1,
) -> Optional[dict[str, Any]]:
    """Compute Voronoi diagram for all players in a frame.

    Args:
        frame: Frame dictionary with tracking data
        player_map: Player ID to metadata mapping
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        rm_team_id: Real Madrid team ID for coloring

    Returns:
        Dictionary with Voronoi diagram data (regions, vertices, team colors),
        or None if insufficient players (<3)
    """
    from scipy.spatial import Voronoi

    positions = []
    team_ids = []

    # Include ALL players (detected and extrapolated) for complete Voronoi diagram
    for p in frame.get("player_data", []):
        x, y = p.get("x"), p.get("y")
        if x is not None and y is not None:
            pid = int(p["player_id"])
            info = player_map.get(pid)
            p_team_id = int(info.get("team_id") or -999) if info else -999
            positions.append((float(x), float(y)))
            team_ids.append(p_team_id)

    if len(positions) < 3:
        return None

    try:
        points = np.array(positions)
        vor = Voronoi(points)

        # Clip Voronoi regions to pitch bounds
        half_length = pitch_length / 2
        half_width = pitch_width / 2

        return {
            "voronoi": vor,
            "team_ids": team_ids,
            "points": points,
            "rm_team_id": rm_team_id,
            "bounds": (-half_length, half_length, -half_width, half_width),
        }
    except (ValueError, RuntimeError):
        return None


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    Args:
        vor: Voronoi diagram from scipy.spatial.Voronoi
        radius: Distance to 'points at infinity'

    Returns:
        regions: List of vertices for each Voronoi region
        vertices: Coordinates of the Voronoi vertices
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def create_voronoi_traces(
    voronoi_data: dict[str, Any],
) -> list[go.Scatter]:
    """Create Plotly traces for team-colored Voronoi regions.

    Args:
        voronoi_data: Dictionary with voronoi, team_ids, points, rm_team_id, and bounds

    Returns:
        List of Plotly Scatter traces for Voronoi regions colored by team
    """
    try:
        from shapely.geometry import Polygon, box
    except ImportError:
        # Fallback if shapely not installed - return empty list
        return []

    vor = voronoi_data["voronoi"]
    team_ids = voronoi_data["team_ids"]
    rm_team_id = voronoi_data["rm_team_id"]
    bounds = voronoi_data["bounds"]
    points = voronoi_data["points"]

    traces = []
    pitch_box = box(bounds[0], bounds[2], bounds[1], bounds[3])

    # Get finite regions for all points
    try:
        regions, vertices = voronoi_finite_polygons_2d(vor)
    except Exception as e:
        print(f"Voronoi: Failed to compute finite regions - {e}")
        return []

    # Debug: Track regions processed
    regions_processed = 0
    regions_skipped = 0

    # Render each region
    for point_idx, region in enumerate(regions):
        if not region or len(region) < 3:
            regions_skipped += 1
            continue

        try:
            # Get region vertices
            polygon_vertices = [vertices[i] for i in region]
            poly = Polygon(polygon_vertices)

            # Clip to pitch bounds
            clipped = poly.intersection(pitch_box)

            if clipped.is_empty:
                regions_skipped += 1
                continue

            # Handle MultiPolygon case (take largest)
            if hasattr(clipped, 'geoms'):
                clipped = max(clipped.geoms, key=lambda p: p.area)

            if not hasattr(clipped, 'exterior'):
                regions_skipped += 1
                continue

            # Get coordinates
            coords = list(clipped.exterior.coords)
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]

            # Determine team color
            team_id = team_ids[point_idx]
            if team_id == rm_team_id:
                fill_color = "rgba(255, 0, 0, 0.15)"
                line_color = "rgba(255, 0, 0, 0.3)"
            else:
                fill_color = "rgba(0, 80, 200, 0.15)"
                line_color = "rgba(0, 80, 200, 0.3)"

            traces.append(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ))
            regions_processed += 1
        except Exception as e:
            # Skip problematic regions
            regions_skipped += 1
            continue

    # Debug output
    print(f"Voronoi: {len(points)} total players, {regions_processed} regions rendered, {regions_skipped} skipped")

    return traces


def compute_movement_vectors(
    frames: list[dict[str, Any]],
    frame_idx: int,
    player_map: dict[int, dict[str, Any]],
    lookback: int = 3,
) -> list[tuple[float, float, float, float, int]]:
    """Compute movement vectors for players based on recent positions.

    Args:
        frames: List of all frames
        frame_idx: Current frame index
        player_map: Player ID to metadata mapping
        lookback: Number of frames to look back for computing velocity

    Returns:
        List of (x, y, dx, dy, team_id) tuples representing movement vectors
    """
    if frame_idx >= len(frames) or len(frames) < 2:
        return []

    # Use actual lookback or fall back to first available frame
    actual_lookback = min(lookback, frame_idx)
    if actual_lookback == 0:
        # For frame 0, compare to frame 1 (future frame)
        if len(frames) < 2:
            return []
        current_frame = frames[0]
        past_frame = frames[1]
        # Reverse direction since we're looking forward
        reverse_direction = True
    else:
        current_frame = frames[frame_idx]
        past_frame = frames[frame_idx - actual_lookback]
        reverse_direction = False

    # Build position maps by player_id
    curr_pos_map = {}
    for p in current_frame.get("player_data", []):
        pid = int(p["player_id"])
        curr_pos_map[pid] = p

    past_pos_map = {}
    for p in past_frame.get("player_data", []):
        pid = int(p["player_id"])
        past_pos_map[pid] = p

    vectors = []

    for pid, curr_pos in curr_pos_map.items():
        # Include both detected and extrapolated players
        past_pos = past_pos_map.get(pid)
        if not past_pos:
            continue

        x1, y1 = past_pos.get("x"), past_pos.get("y")
        x2, y2 = curr_pos.get("x"), curr_pos.get("y")

        if None in (x1, y1, x2, y2):
            continue

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)

        # Reverse direction if looking at future frame
        if reverse_direction:
            dx, dy = -dx, -dy

        # Only show vectors with meaningful movement (very low threshold for debugging)
        if math.hypot(dx, dy) > 0.05:  # At least 0.05m movement (5cm)
            info = player_map.get(pid)
            p_team_id = int(info.get("team_id") or -999) if info else -999
            vectors.append((float(x2), float(y2), dx, dy, p_team_id))

    # Debug output - always print to help diagnose issues
    print(f"Movement vectors: Frame {frame_idx}, total frames={len(frames)}, lookback={lookback}, actual_lookback={actual_lookback if 'actual_lookback' in locals() else 'N/A'}, vectors={len(vectors)}")

    if vectors:
        print(f"  → Sample vector: position=({vectors[0][0]:.1f}, {vectors[0][1]:.1f}), direction=({vectors[0][2]:.2f}, {vectors[0][3]:.2f})")
    else:
        print(f"  → No vectors generated (movement < 0.2m threshold or no player matches)")

    return vectors


def render_dashboard(idx: pd.DataFrame, features_df: pd.DataFrame) -> None:
    """Render aggregate analytics dashboard across all build-ups."""
    st.header("Dashboard - Aggregate Analytics")

    if features_df.empty:
        st.warning("No features available. Run: python main.py")
        st.code("python -m src.features.feature_engineering --processed-root data/processed/rm_pressing", language="bash")
        return

    # Summary Statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Build-Ups", len(idx))
        st.metric("Unique Games", idx['game_id'].nunique())

    with col2:
        if 'pressure_frames_ratio' in features_df.columns:
            avg_pressure = features_df['pressure_frames_ratio'].mean() * 100
            st.metric("Avg Pressure Intensity", f"{avg_pressure:.1f}%")
        if 't_first_pressure_s' in features_df.columns:
            avg_first = features_df['t_first_pressure_s'].mean()
            st.metric("Avg Time to Pressure", f"{avg_first:.1f}s")

    with col3:
        if 'rm_width_mean_m' in features_df.columns:
            avg_width = features_df['rm_width_mean_m'].mean()
            st.metric("Avg Team Width", f"{avg_width:.1f}m")
        if 'rm_hull_area_mean_m2' in features_df.columns:
            avg_area = features_df['rm_hull_area_mean_m2'].mean()
            st.metric("Avg Hull Area", f"{avg_area:.0f}m²")

    with col4:
        if 'goal_kick_type' in features_df.columns:
            short_pct = (features_df['goal_kick_type'] == 'short').sum() / len(features_df) * 100
            st.metric("Short Kicks", f"{short_pct:.1f}%")
        unique_opponents = idx['opponent_team_name'].nunique()
        st.metric("Unique Opponents", unique_opponents)

    st.divider()

    # Distribution Charts
    st.subheader("Feature Distributions")

    tab1, tab2, tab3 = st.tabs(["Pressure Metrics", "Compactness Metrics", "Kick Types"])

    with tab1:
        if 'pressure_frames_ratio' in features_df.columns:
            col1, col2 = st.columns(2)

            with col1:
                fig_pressure = go.Figure()
                fig_pressure.add_trace(go.Histogram(
                    x=features_df['pressure_frames_ratio'] * 100,
                    nbinsx=20,
                    name="Pressure Intensity",
                    marker_color='indianred'
                ))
                fig_pressure.update_layout(
                    title="Pressure Intensity Distribution",
                    xaxis_title="Pressure Frames Ratio (%)",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_pressure, use_container_width=True)

            with col2:
                if 't_first_pressure_s' in features_df.columns:
                    fig_first = go.Figure()
                    fig_first.add_trace(go.Histogram(
                        x=features_df['t_first_pressure_s'].dropna(),
                        nbinsx=20,
                        name="Time to First Pressure",
                        marker_color='steelblue'
                    ))
                    fig_first.update_layout(
                        title="Time to First Pressure Distribution",
                        xaxis_title="Seconds",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_first, use_container_width=True)

    with tab2:
        if 'rm_width_mean_m' in features_df.columns:
            col1, col2 = st.columns(2)

            with col1:
                fig_width = go.Figure()
                fig_width.add_trace(go.Histogram(
                    x=features_df['rm_width_mean_m'],
                    nbinsx=20,
                    name="Team Width",
                    marker_color='seagreen'
                ))
                fig_width.update_layout(
                    title="Team Width Distribution",
                    xaxis_title="Width (meters)",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_width, use_container_width=True)

            with col2:
                if 'rm_hull_area_mean_m2' in features_df.columns:
                    fig_hull = go.Figure()
                    fig_hull.add_trace(go.Histogram(
                        x=features_df['rm_hull_area_mean_m2'],
                        nbinsx=20,
                        name="Hull Area",
                        marker_color='mediumpurple'
                    ))
                    fig_hull.update_layout(
                        title="Convex Hull Area Distribution",
                        xaxis_title="Area (m²)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_hull, use_container_width=True)

    with tab3:
        if 'goal_kick_type' in features_df.columns:
            kick_counts = features_df['goal_kick_type'].value_counts()
            fig_kicks = go.Figure(data=[go.Pie(
                labels=kick_counts.index,
                values=kick_counts.values,
                hole=0.3
            )])
            fig_kicks.update_layout(
                title="Goal Kick Types",
                height=400
            )
            st.plotly_chart(fig_kicks, use_container_width=True)

            # Kick distance box plot
            if 'gk_kick_distance_m' in features_df.columns:
                fig_dist = go.Figure()
                for kick_type in features_df['goal_kick_type'].unique():
                    data = features_df[features_df['goal_kick_type'] == kick_type]['gk_kick_distance_m']
                    fig_dist.add_trace(go.Box(y=data, name=kick_type))

                fig_dist.update_layout(
                    title="Kick Distance by Type",
                    yaxis_title="Distance (meters)",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # Correlation Analysis
    st.subheader("Feature Correlations")

    numeric_features = ['pressure_frames_ratio', 'rm_width_mean_m', 'rm_length_mean_m',
                        'rm_hull_area_mean_m2', 'gk_kick_distance_m', 't_first_pressure_s']
    available_features = [f for f in numeric_features if f in features_df.columns]

    if len(available_features) >= 2:
        corr_data = features_df[available_features].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        fig_corr.update_layout(
            title="Feature Correlation Matrix",
            height=500,
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_corr, use_container_width=True)


def main() -> None:
    st.title("Real Madrid Pressing Analysis")
    st.caption("Data: `data/processed/rm_pressing/`")

    # Common configuration (before tabs)
    with st.sidebar:
        st.header("Configuration")
        processed_root = st.text_input("Processed root", DEFAULT_PROCESSED_ROOT)
        raw_root = st.text_input("Raw data root (meta)", DEFAULT_RAW_ROOT)
        features_root = st.text_input("Features root", DEFAULT_FEATURES_ROOT)

        c3, c4, c5 = st.columns(3)
        min_players = c3.number_input("Min players shown/frame", 0, 22, int(DEFAULT_MIN_PLAYERS), 1)
        include_extrapolated = c4.checkbox("Include extrapolated positions", value=bool(DEFAULT_INCLUDE_EXTRAPOLATED))
        fill_missing_frames = c5.checkbox("Fill missing frames (interpolate)", value=False)

        c6, c7, c8, c9 = st.columns(4)
        max_gap_fill_s = c6.number_input("Max gap to fill (s)", 0.0, 30.0, 2.0, 0.5)
        show_rm_rings = c7.checkbox("Red rings on Real Madrid", value=True)
        rm_ring_size = c8.slider("RM ring size", 18, 42, 28, 1)
        rm_ring_width = c9.slider("RM ring width", 1, 6, 2, 1)

        st.markdown("##### Pressing Network Options")
        c10, c11 = st.columns(2)
        pressure_threshold_m = c10.slider("Pressure threshold (m)", 3.0, 10.0, float(DEFAULT_PRESSURE_THRESHOLD), 0.5)
        show_trigger = c11.checkbox("Highlight trigger player", value=True)

        # Always show pressing network and pressure zones
        show_pressing_network = True
        show_pressure_zones = True

        st.markdown("##### Statistical Overlays")
        c14, c15, c16, c17 = st.columns(4)
        show_convex_hull = c14.checkbox("Show RM hull", value=False)
        show_opp_hull = c15.checkbox("Show Opp hull", value=False)
        show_voronoi = c16.checkbox("Show Voronoi diagram", value=False)
        show_vectors = c17.checkbox("Show movement vectors", value=False)

        show_timeline = st.checkbox("Show pressure timeline", value=False)

        phase = st.selectbox("Phase", ["Saved window", "Ready → kick", "Kick → end"], index=0)

    index_path = _resolve_user_path(processed_root) / "index.parquet"
    if not index_path.exists():
        st.error("Processed index not found.")
        st.code(f'python extraction.py --out-dir "{processed_root}" --full', language="bash")
        st.code(f'python extraction.py --out-dir "{processed_root}" --match-id 2014987', language="bash")
        return

    idx = load_processed_index(processed_root)
    if idx.empty:
        st.warning("Processed index is empty.")
        return

    # Load features
    features_df = load_features(features_root)
    if features_df.empty:
        st.warning("Features not found. Run: python main.py")

    # Create tab navigation
    tab1, tab2 = st.tabs(["Viewer", "Comparison"])

    # =======================
    # TAB 1: VIEWER
    # =======================
    with tab1:
        st.subheader("Build-ups")
        st.dataframe(idx, use_container_width=True, hide_index=True)

        ids = idx["build_up_id"].astype(int).tolist()

        def _fmt_build_up(bid: int) -> str:
            r = idx[idx["build_up_id"] == bid].iloc[0]
            return f"{bid:05d}. game {r['game_id']} | P{int(r['period'])} | {r['opponent_team_name']} | ready {r['ready_time']} | kick {r['kick_time']}"

        # Default to build-up 6 (first one with pressure data)
        default_index = 5 if len(ids) > 5 else 0
        pick_id = st.selectbox("Select build-up", ids, index=default_index, format_func=_fmt_build_up)

        # Add helpful note about build-ups without pressure
        if pick_id <= 5:
            st.info("Note: Build-ups 1-5 have no pressure detected due to RM defenders being 11-35m away from the ball carrier. Select build-up 6 or higher to view pressure metrics.")
        row = idx[idx["build_up_id"] == int(pick_id)].iloc[0].to_dict()

        # Get features for this build-up
        features = get_build_up_features(features_df, int(pick_id))

        df_long = load_processed_frames(processed_root, str(row["frames_path"]))

        # Custom time range controls
        use_custom_range = st.checkbox("Use custom time range", value=False)

        if use_custom_range:
            st.caption(f"**Key events:** Ready: {row['ready_time']} | Kick: {row['kick_time']}")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                custom_start = st.text_input("Start time (MM:SS.S)", value=str(row["window_start"]))
            with col_t2:
                custom_end = st.text_input("End time (MM:SS.S)", value=str(row["window_end"]))
            t0 = custom_start
            t1 = custom_end
        else:
            if phase.startswith("Ready"):
                t0 = str(row["ready_time"])
                t1 = str(row["kick_time"])
            elif phase.startswith("Kick"):
                t0 = str(row["kick_time"])
                t1 = str(row["window_end"])
            else:
                t0 = str(row["window_start"])
                t1 = str(row["window_end"])

        df_long = df_long[(df_long["time"] >= t0) & (df_long["time"] <= t1)].copy()
        frames = build_window_frames_from_long_df(
            df=df_long,
            period=int(row["period"]),
            min_players=int(min_players),
            include_extrapolated=bool(include_extrapolated),
            require_ball_detected=False,
        )

        fps = estimate_fps_from_frames(frames, default_fps=10)
        if len(frames) > 1:
            secs = [time_to_seconds(str(fr.get("timestamp"))) for fr in frames]
            max_gap = max(secs[i + 1] - secs[i] for i in range(len(secs) - 1))
        else:
            max_gap = 0.0

        st.caption(f"Estimated FPS: {int(fps)} | Frames: {len(frames)} | Max gap: {max_gap:.1f}s")

        if bool(fill_missing_frames):
            frames = fill_frame_gaps_linear(frames=frames, fps=int(fps), max_gap_seconds=float(max_gap_fill_s))

        meta = load_meta(str(row["game_id"]), raw_root)
        player_map = build_player_mapping(meta)

        # Identify trigger player
        trigger_info = None
        if show_trigger and frames:
            from src.viz.pressing_network import identify_trigger_player

            trigger_info = identify_trigger_player(
                frames=frames,
                player_map=player_map,
                rm_team_id=int(row["rm_team_id"]),
                pressure_threshold_m=pressure_threshold_m,
                kick_time=str(row["kick_time"]),
            )

        # Compute live pressure metrics so the panel matches the current window + threshold
        live_pressure = compute_pressure_metrics_from_frames(
            frames=frames,
            player_map=player_map,
            rm_team_id=int(row["rm_team_id"]),
            opponent_team_id=int(row["opponent_team_id"]),
            pressure_threshold_m=float(pressure_threshold_m),
        )
        features_for_panel = dict(features or {})
        features_for_panel.update(live_pressure)

        # Render info panel (works even without features.parquet)
        render_info_panel(
            features=features_for_panel,
            trigger_info=trigger_info,
            player_map=player_map,
        )

        with st.expander("Selected build-up details", expanded=False):
            st.json(row)

        title = f"{row['rm_team_name']} pressing – {row['opponent_team_name']} build-up"

        # Create animation with loading indicator
        with st.spinner("Creating animation with overlays..."):
            fig = create_animation_figure(
                frames=frames,
                meta=meta,
                player_map=player_map,
                rm_team_id=int(row["rm_team_id"]),
                opponent_team_id=int(row["opponent_team_id"]),
                title_line=title,
                kick_time=str(row["kick_time"]),
                fps=int(fps),
                show_rm_rings=bool(show_rm_rings),
                rm_ring_size=int(rm_ring_size),
                rm_ring_width=int(rm_ring_width),
                show_pressing_network=bool(show_pressing_network),
                show_pressure_zones=bool(show_pressure_zones),
                pressure_threshold_m=float(pressure_threshold_m),
                trigger_info=trigger_info if show_trigger else None,
                show_convex_hull=bool(show_convex_hull),
                show_opp_hull=bool(show_opp_hull),
                show_voronoi=bool(show_voronoi),
                show_vectors=bool(show_vectors),
            )

        if fig is None:
            st.warning("No frames for this selection (try lowering Min players shown/frame).")
            return

        st.plotly_chart(fig, use_container_width=True)

        # Event markers - show which frames correspond to key events
        if frames and len(frames) > 1:
            ready_time = str(row["ready_time"])
            kick_time = str(row["kick_time"])

            # Find frame indices for key events
            ready_frame_idx = None
            kick_frame_idx = None

            for i, fr in enumerate(frames):
                fr_time = str(fr.get("timestamp", ""))
                if fr_time == ready_time:
                    ready_frame_idx = i
                if fr_time == kick_time:
                    kick_frame_idx = i

            # Display event markers
            event_info = []
            if ready_frame_idx is not None:
                event_info.append(f"**Ready**: Frame {ready_frame_idx + 1}/{len(frames)} ({ready_time})")
            if kick_frame_idx is not None:
                event_info.append(f"**Kick**: Frame {kick_frame_idx + 1}/{len(frames)} ({kick_time})")

            if event_info:
                st.info(" | ".join(event_info))

        # Render pressure timeline if enabled
        if show_timeline:
            timeline_fig = render_pressure_timeline(
                frames=frames,
                rm_team_id=int(row["rm_team_id"]),
                opponent_team_id=int(row["opponent_team_id"]),
                player_map=player_map,
                pressure_threshold_m=float(pressure_threshold_m),
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

    # =======================
    # TAB 2: COMPARISON
    # =======================
    with tab2:
        st.subheader("Multi-Build-Up Comparison")

        # Filtering options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_opponent = st.multiselect(
                "Filter by opponent",
                options=sorted(idx["opponent_team_name"].unique()),
                default=[]
            )
        with col2:
            filter_period = st.multiselect(
                "Filter by period",
                options=sorted(idx["period"].unique()),
                default=[]
            )
        with col3:
            if not features_df.empty and "goal_kick_type" in features_df.columns:
                filter_kick_type = st.multiselect(
                    "Filter by kick type",
                    options=["short", "long"],
                    default=[]
                )
            else:
                filter_kick_type = []
        with col4:
            num_comparisons = st.selectbox("Number of build-ups", [2, 3, 4], index=0)

        # Apply filters
        filtered_idx = idx.copy()
        if filter_opponent:
            filtered_idx = filtered_idx[filtered_idx["opponent_team_name"].isin(filter_opponent)]
        if filter_period:
            filtered_idx = filtered_idx[filtered_idx["period"].isin(filter_period)]
        if filter_kick_type and not features_df.empty:
            # Join with features to filter by kick type
            filtered_idx = filtered_idx.merge(
                features_df[["build_up_id", "goal_kick_type"]],
                on="build_up_id",
                how="inner"
            )
            filtered_idx = filtered_idx[filtered_idx["goal_kick_type"].isin(filter_kick_type)]

        if filtered_idx.empty:
            st.warning("No build-ups match the selected filters.")
        else:
            st.caption(f"{len(filtered_idx)} build-ups available for comparison")

            # Multi-select for build-ups
            ids_available = filtered_idx["build_up_id"].astype(int).tolist()

            def _fmt_comparison(bid: int) -> str:
                r = filtered_idx[filtered_idx["build_up_id"] == bid].iloc[0]
                return f"{bid:05d} | {r['opponent_team_name']} | P{int(r['period'])}"

            selected_ids = st.multiselect(
                f"Select {num_comparisons} build-ups to compare",
                ids_available,
                format_func=_fmt_comparison,
                max_selections=num_comparisons
            )

            if len(selected_ids) == num_comparisons:
                # Create comparison grid
                if num_comparisons == 2:
                    cols = st.columns(2)
                elif num_comparisons == 3:
                    cols = st.columns(3)
                else:  # 4
                    cols = st.columns(2)

                for i, bid in enumerate(selected_ids):
                    row_data = filtered_idx[filtered_idx["build_up_id"] == int(bid)].iloc[0].to_dict()
                    meta = load_meta(str(row_data["game_id"]), raw_root)
                    player_map = build_player_mapping(meta)

                    df_long = load_processed_frames(processed_root, str(row_data["frames_path"]))
                    t0 = str(row_data["window_start"])
                    t1 = str(row_data["window_end"])
                    df_long = df_long[(df_long["time"] >= t0) & (df_long["time"] <= t1)].copy()

                    frames = build_window_frames_from_long_df(
                        df=df_long,
                        period=int(row_data["period"]),
                        min_players=int(min_players),
                        include_extrapolated=bool(include_extrapolated),
                        require_ball_detected=False,
                    )

                    if frames:
                        fps = estimate_fps_from_frames(frames, default_fps=10)
                        title = f"{bid:05d} - {row_data['opponent_team_name']}"

                        fig = create_animation_figure(
                            frames=frames,
                            meta=meta,
                            player_map=player_map,
                            rm_team_id=int(row_data["rm_team_id"]),
                            opponent_team_id=int(row_data["opponent_team_id"]),
                            title_line=title,
                            kick_time=str(row_data["kick_time"]),
                            fps=int(fps),
                            show_rm_rings=bool(show_rm_rings),
                            rm_ring_size=int(rm_ring_size),
                            rm_ring_width=int(rm_ring_width),
                            show_pressing_network=bool(show_pressing_network),
                            show_pressure_zones=bool(show_pressure_zones),
                            pressure_threshold_m=float(pressure_threshold_m),
                            trigger_info=None,  # Simplified for comparison
                            show_convex_hull=bool(show_convex_hull),
                            show_opp_hull=bool(show_opp_hull),
                            show_voronoi=bool(show_voronoi),
                            show_vectors=bool(show_vectors),
                        )

                        if num_comparisons == 4:
                            # Use 2x2 grid for 4 comparisons
                            col_idx = i % 2
                            with cols[col_idx]:
                                st.markdown(f"**{title}**")
                                st.plotly_chart(fig, use_container_width=True, key=f"comp_{bid}")
                        else:
                            with cols[i]:
                                st.markdown(f"**{title}**")
                                st.plotly_chart(fig, use_container_width=True, key=f"comp_{bid}")

                # Comparative statistics table
                if not features_df.empty:
                    st.markdown("### Comparison")
                    comp_features = features_df[features_df["build_up_id"].isin(selected_ids)].copy()
                    if not comp_features.empty:
                        # Select key metrics for comparison
                        metrics_to_show = [
                            "build_up_id",
                            "t_first_pressure_s",
                            "pressure_frames_ratio",
                            "rm_width_mean_m",
                            "rm_length_mean_m",
                            "rm_hull_area_mean_m2",
                            "goal_kick_type",
                            "gk_kick_distance_m",
                        ]
                        available_metrics = [m for m in metrics_to_show if m in comp_features.columns]
                        comp_table = comp_features[available_metrics].set_index("build_up_id")
                        st.dataframe(comp_table, use_container_width=True)
            elif selected_ids:
                st.info(f"Select {num_comparisons - len(selected_ids)} more build-up(s)")


if __name__ == "__main__":
    main()
