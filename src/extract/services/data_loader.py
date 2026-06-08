"""Data loading utilities for SkillCorner football tracking data.

This module handles loading and converting various data formats including:
- Metadata (JSON files with match/player information)
- Dynamic event data (Parquet files)
- Tracking data (JSON or Parquet formats)
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd


# Heuristic: a full match long-format parquet should be tens of MB
MIN_TRACKING_PARQUET_BYTES = 5_000_000


def list_game_ids_by_suffix(folder: Path, suffix: str) -> list[str]:
    """List all game IDs in a folder by file suffix.

    Args:
        folder: Directory path to search
        suffix: File suffix to match (e.g., ".json", ".parquet")

    Returns:
        Sorted list of game IDs (file stems)
    """
    if not folder.exists():
        return []
    return sorted([p.stem for p in folder.glob(f"*{suffix}")])


def list_full_game_ids(data_root: Path) -> list[str]:
    """List all game IDs that have complete data (meta + dynamic + tracking).

    A "full" game has:
    - meta/<id>.json
    - dynamic/<id>.parquet
    - tracking data (either tracking_parquet/<id>.parquet or tracking/<id>.json)

    Args:
        data_root: Root directory containing data subdirectories

    Returns:
        Sorted list of game IDs with complete data
    """
    meta_ids = set(list_game_ids_by_suffix(data_root / "meta", ".json"))
    dyn_ids = set(list_game_ids_by_suffix(data_root / "dynamic", ".parquet"))
    tracking_parquet_ids = set(list_game_ids_by_suffix(data_root / "tracking_parquet", ".parquet"))
    tracking_json_ids = set(list_game_ids_by_suffix(data_root / "tracking", ".json"))
    tracking_ids = tracking_parquet_ids | tracking_json_ids
    return sorted(meta_ids & dyn_ids & tracking_ids)


def load_meta(game_id: str, data_root: Path) -> dict[str, Any]:
    """Load metadata JSON file for a game.

    Args:
        game_id: Match identifier
        data_root: Root directory containing meta/ subdirectory

    Returns:
        Dictionary containing match metadata
    """
    path = data_root / "meta" / f"{game_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dynamic(game_id: str, data_root: Path) -> pd.DataFrame:
    """Load dynamic event data parquet file for a game.

    Args:
        game_id: Match identifier
        data_root: Root directory containing dynamic/ subdirectory

    Returns:
        DataFrame containing event data
    """
    path = data_root / "dynamic" / f"{game_id}.parquet"
    return pd.read_parquet(path)


def load_tracking_parquet(game_id: str, data_root: Path) -> pd.DataFrame:
    """Load pre-converted tracking parquet file.

    Args:
        game_id: Match identifier
        data_root: Root directory containing tracking_parquet/ subdirectory

    Returns:
        DataFrame with columns: match_id, time, frame, period, player_id,
                                is_detected, is_ball, x, y
    """
    path = data_root / "tracking_parquet" / f"{game_id}.parquet"
    return pd.read_parquet(path)


def convert_tracking_json_to_long_df(
    *,
    game_id: str,
    data_root: Path,
    cache_to_parquet: bool
) -> pd.DataFrame:
    """Convert SkillCorner tracking JSON to long-format DataFrame.

    Converts frames list into long-format rows compatible with tracking_parquet format.
    Ball uses player_id=-1. Only includes frames with valid timestamp, period, and frame number.

    Args:
        game_id: Match identifier
        data_root: Root directory containing tracking/ subdirectory
        cache_to_parquet: If True, save converted data to tracking_parquet/

    Returns:
        DataFrame with columns: match_id, time, frame, period, player_id,
                                is_detected, is_ball, x, y
    """
    tracking_path = data_root / "tracking" / f"{game_id}.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        tracking = json.load(f)

    rows: list[dict[str, Any]] = []
    match_id = int(game_id)

    for fr in tracking:
        ts = fr.get("timestamp")
        period = fr.get("period")
        frame = fr.get("frame")
        if ts is None or period is None or frame is None:
            continue

        try:
            frame_i = int(frame)
            period_i = int(period)
        except (TypeError, ValueError):
            continue

        # Process ball data
        b = fr.get("ball_data") or {}
        if b:
            bx = b.get("x")
            by = b.get("y")
            bis = b.get("is_detected")
            if bx is not None and by is not None:
                rows.append(
                    {
                        "match_id": match_id,
                        "time": str(ts),
                        "frame": frame_i,
                        "period": period_i,
                        "player_id": -1,
                        "is_detected": bool(bis) if bis is not None else False,
                        "is_ball": True,
                        "x": float(bx),
                        "y": float(by),
                    }
                )

        # Process player data
        for p in fr.get("player_data", []) or []:
            pid = p.get("player_id")
            if pid is None:
                continue

            x = p.get("x")
            y = p.get("y")
            if x is None or y is None:
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "time": str(ts),
                    "frame": frame_i,
                    "period": period_i,
                    "player_id": int(pid),
                    "is_detected": bool(p.get("is_detected")),
                    "is_ball": False,
                    "x": float(x),
                    "y": float(y),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["period", "frame", "is_ball"], ignore_index=True)

    if cache_to_parquet:
        out_path = data_root / "tracking_parquet" / f"{game_id}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    # Clean up memory
    del tracking
    del rows
    gc.collect()
    return df


def load_tracking_auto(
    *,
    game_id: str,
    data_root: Path,
    cache_json_to_parquet: bool,
    rebuild_tracking_parquet: bool
) -> pd.DataFrame:
    """Automatically load tracking data from parquet or JSON.

    Tries to load from tracking_parquet/ first. If file doesn't exist or is too small
    (indicating incomplete cache), falls back to converting from tracking/ JSON.

    Args:
        game_id: Match identifier
        data_root: Root directory containing tracking data
        cache_json_to_parquet: If True, cache converted JSON to parquet
        rebuild_tracking_parquet: If True, force rebuild from JSON even if parquet exists

    Returns:
        DataFrame with tracking data in long format

    Raises:
        FileNotFoundError: If no tracking data found in either format
    """
    parquet_path = data_root / "tracking_parquet" / f"{game_id}.parquet"
    if parquet_path.exists() and not bool(rebuild_tracking_parquet):
        try:
            if parquet_path.stat().st_size < MIN_TRACKING_PARQUET_BYTES and (data_root / "tracking" / f"{game_id}.json").exists():
                # Likely an incomplete cache (e.g. ball-only). Rebuild from JSON.
                rebuild_tracking_parquet = True
        except OSError:
            pass
        if not bool(rebuild_tracking_parquet):
            return pd.read_parquet(parquet_path)

    json_path = data_root / "tracking" / f"{game_id}.json"
    if json_path.exists():
        return convert_tracking_json_to_long_df(
            game_id=game_id,
            data_root=data_root,
            cache_to_parquet=bool(cache_json_to_parquet),
        )

    raise FileNotFoundError(f"No tracking found for {game_id} under tracking_parquet/ or tracking/")
