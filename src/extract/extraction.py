"""Main extraction orchestrator for Real Madrid pressing analysis.

This script identifies opponent goal-kick build-up moments from SkillCorner data and saves short
tracking windows to the output directory (default: `data/processed/rm_pressing/`).

Inputs (under `--data-root`, default `data/raw/RealMadrid`):
  - `meta/<match_id>.json`
  - `dynamic/<match_id>.parquet`
  - tracking data either as:
      - `tracking_parquet/<match_id>.parquet` (preferred), or
      - `tracking/<match_id>.json` (will be converted on the fly)

Outputs (under `--out-dir`, default `data/processed/rm_pressing`):
  - `index.parquet` (one row per extracted build-up)
  - `frames/build_up_XXXXXXX.parquet` (long-format tracking rows for the saved window)
  - `params.json` (extractor settings; does not store absolute paths)

Usage:
    python extraction.py                    # Process default 5 games
    python extraction.py --full             # Process all available games
    python extraction.py 2014987 2016604    # Process specific games
    python extraction.py --match-id 2014987 # Alternative syntax for specific games
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.build_up_detector import detect_build_up_from_reference, diagnose_build_up_from_reference
from services.data_loader import (
    list_full_game_ids,
    load_dynamic,
    load_meta,
    load_tracking_auto,
)
from services.goal_kick_detector import filter_goal_kick_refs
from services.team_utils import find_real_madrid_and_opponent
from services.time_utils import add_seconds_to_time


# Default paths
APP_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = APP_ROOT / "data" / "raw" / "RealMadrid"
DEFAULT_PROCESSED_ROOT = APP_ROOT / "data" / "processed" / "rm_pressing"

# Default game set
DEFAULT_FIVE_GAMES = ["2014987", "2016604", "2017232", "2017683", "2018396"]

# Detection parameters - time windows
DEFAULT_LOOKBACK_SECONDS = 90
DEFAULT_WINDOW_AFTER_SECONDS = 10
DEFAULT_PRE_KICK_SECONDS = 2.0
DEFAULT_START_FROM_READY = False
DEFAULT_MAX_SECONDS_BEFORE_REF = 90.0

# Detection parameters - geometry
DEFAULT_GOAL_AREA_DEPTH_M = 5.5
DEFAULT_GOAL_AREA_HALF_WIDTH_M = 10.0
DEFAULT_GOAL_AREA_X_MARGIN_M = 1.0
DEFAULT_GK_BALL_DISTANCE_M = 6.0

# Detection parameters - kick detection
DEFAULT_KICK_DISPLACEMENT_M = 1.0
DEFAULT_KICK_CONFIRM_FRAMES = 5

# Detection parameters - ready state
DEFAULT_READY_MIN_GK_BALL_DISTANCE_M = 2.0
DEFAULT_READY_MAX_GK_BALL_DISTANCE_M = 5.0
DEFAULT_READY_STABLE_FRAMES = 3
DEFAULT_READY_BALL_STEP_EPS_M = 0.5
DEFAULT_READY_GK_STEP_EPS_M = 1.0


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def safe_relpath(path: Path, *, base: Path) -> str | None:
    """Get relative path string, or None if not relative to base."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return None


def resolve_user_path(path: Path) -> Path:
    """Resolve a user-provided path.

    - Absolute paths are returned as-is.
    - Relative paths are interpreted as relative to the repository root (APP_ROOT).
    """
    if path.is_absolute():
        return path
    return (APP_ROOT / path).resolve()


def run_extractor_to_disk(
    *,
    data_root: Path,
    out_dir: Path,
    game_ids: list[str],
    lookback_seconds: int,
    window_after_seconds: int,
    pre_kick_seconds: float,
    start_from_ready: bool,
    max_seconds_before_ref: float,
    goal_area_depth_m: float,
    goal_area_half_width_m: float,
    goal_area_x_margin_m: float,
    gk_ball_distance_m: float,
    kick_displacement_m: float,
    kick_confirm_frames: int,
    cache_tracking_parquet: bool,
    rebuild_tracking_parquet: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run the extraction process and save results to disk.

    Args:
        data_root: Root directory containing raw data
        out_dir: Output directory for processed data
        game_ids: List of game IDs to process
        lookback_seconds: Seconds to look back from goal kick event
        window_after_seconds: Seconds to include after kick/ready time
        pre_kick_seconds: Seconds before kick to include (if not start_from_ready)
        start_from_ready: Start window from ready time instead of kick time
        max_seconds_before_ref: Maximum seconds before event reference to accept
        goal_area_depth_m: Goal area depth from goal line
        goal_area_half_width_m: Goal area half-width
        goal_area_x_margin_m: Extra margin for goal area
        gk_ball_distance_m: Max GK-ball distance for setup
        kick_displacement_m: Min displacement to detect kick
        kick_confirm_frames: Frames to confirm kick
        cache_tracking_parquet: Cache converted JSON to parquet
        rebuild_tracking_parquet: Force rebuild from JSON
        verbose: Print progress information

    Returns:
        Dictionary with extraction results summary
    """
    # Setup output directories
    frames_dir = out_dir / "frames"
    ensure_dir(frames_dir)

    # Clean up old frame files
    for old in frames_dir.glob("build_up_*.parquet"):
        try:
            old.unlink()
        except OSError:
            pass

    seen_keys: set[tuple[str, int, str]] = set()
    index_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    build_up_id = 0
    started_at = datetime.now().isoformat(timespec="seconds")

    # Process each game
    for gid in game_ids:
        meta = load_meta(gid, data_root)
        dyn = load_dynamic(gid, data_root)
        tracking = load_tracking_auto(
            game_id=gid,
            data_root=data_root,
            cache_json_to_parquet=bool(cache_tracking_parquet),
            rebuild_tracking_parquet=bool(rebuild_tracking_parquet),
        )

        rm_team, opp_team = find_real_madrid_and_opponent(meta)
        refs = filter_goal_kick_refs(dyn, int(opp_team["id"]), gid)
        competition_edition = meta.get("competition_edition") or {}
        competition = competition_edition.get("competition") or {}
        season = competition_edition.get("season") or {}
        season_name = str(season.get("name") or "")
        season_start_year = season.get("start_year")
        season_end_year = season.get("end_year")
        competition_name = str(competition.get("name") or "")
        competition_area = str(competition.get("area") or "")

        if verbose:
            print(f"[{gid}] refs={len(refs)} | RM={rm_team['name']} | OPP={opp_team['name']}", file=sys.stderr)

        # Process each goal kick reference
        for ref in refs:
            diag = diagnose_build_up_from_reference(
                tracking_df=tracking,
                meta=meta,
                goal_kick_ref=ref,
                rm_team=rm_team,
                opponent_team=opp_team,
                lookback_seconds=int(lookback_seconds),
                goal_area_depth_m=float(goal_area_depth_m),
                goal_area_half_width_m=float(goal_area_half_width_m),
                goal_area_x_margin_m=float(goal_area_x_margin_m),
                gk_ball_distance_m=float(gk_ball_distance_m),
                kick_displacement_m=float(kick_displacement_m),
                kick_confirm_frames=int(kick_confirm_frames),
                ready_min_dist_m=float(DEFAULT_READY_MIN_GK_BALL_DISTANCE_M),
                ready_max_dist_m=float(DEFAULT_READY_MAX_GK_BALL_DISTANCE_M),
                ready_stable_frames=int(DEFAULT_READY_STABLE_FRAMES),
                ready_ball_step_eps_m=float(DEFAULT_READY_BALL_STEP_EPS_M),
                ready_gk_step_eps_m=float(DEFAULT_READY_GK_STEP_EPS_M),
                min_setup_frames=10,
                setup_gap_frames=3,
            )
            diag["season_name"] = season_name
            diag["season_start_year"] = season_start_year
            diag["season_end_year"] = season_end_year
            diag["competition_name"] = competition_name
            diag["competition_area"] = competition_area
            bu = detect_build_up_from_reference(
                tracking_df=tracking,
                meta=meta,
                goal_kick_ref=ref,
                rm_team=rm_team,
                opponent_team=opp_team,
                lookback_seconds=int(lookback_seconds),
                goal_area_depth_m=float(goal_area_depth_m),
                goal_area_half_width_m=float(goal_area_half_width_m),
                goal_area_x_margin_m=float(goal_area_x_margin_m),
                gk_ball_distance_m=float(gk_ball_distance_m),
                kick_displacement_m=float(kick_displacement_m),
                kick_confirm_frames=int(kick_confirm_frames),
                ready_min_dist_m=float(DEFAULT_READY_MIN_GK_BALL_DISTANCE_M),
                ready_max_dist_m=float(DEFAULT_READY_MAX_GK_BALL_DISTANCE_M),
                ready_stable_frames=int(DEFAULT_READY_STABLE_FRAMES),
                ready_ball_step_eps_m=float(DEFAULT_READY_BALL_STEP_EPS_M),
                ready_gk_step_eps_m=float(DEFAULT_READY_GK_STEP_EPS_M),
                min_setup_frames=10,
                setup_gap_frames=3,
                debug=[],
            )
            if not bu:
                diagnostics_rows.append(diag)
                continue

            # Filter by time threshold
            if float(bu.time_before_event_s) > float(max_seconds_before_ref):
                diag["build_up_detected"] = False
                diag["failure_reason"] = "time_before_ref_exceeds_max"
                diagnostics_rows.append(diag)
                continue

            # Deduplicate by (game_id, period, kick_time)
            key = (bu.game_id, int(bu.period), str(bu.kick_time))
            if key in seen_keys:
                diag["build_up_detected"] = False
                diag["failure_reason"] = "duplicate_kick_time"
                diagnostics_rows.append(diag)
                continue
            seen_keys.add(key)

            # Determine time window to extract
            if bool(start_from_ready):
                window_start = bu.ready_time
                window_end = add_seconds_to_time(str(window_start), float(window_after_seconds))
            else:
                window_start = add_seconds_to_time(bu.kick_time, -float(pre_kick_seconds))
                window_end = add_seconds_to_time(bu.kick_time, float(window_after_seconds))

            build_up_id += 1

            # Extract tracking data slice
            df_slice = tracking[
                (tracking["period"] == int(bu.period))
                & (tracking["time"] >= str(window_start))
                & (tracking["time"] <= str(window_end))
            ][["match_id", "time", "frame", "period", "player_id", "is_detected", "is_ball", "x", "y"]].copy()

            # Save tracking slice
            rel_path = f"frames/build_up_{build_up_id:07d}.parquet"
            df_slice.to_parquet(out_dir / rel_path, index=False)

            # Add to index
            index_rows.append(
                {
                    "build_up_id": int(build_up_id),
                    "game_id": str(bu.game_id),
                    "period": int(bu.period),
                    "season_name": season_name,
                    "season_start_year": season_start_year,
                    "season_end_year": season_end_year,
                    "competition_name": competition_name,
                    "competition_area": competition_area,
                    "rm_team_id": int(bu.rm_team_id),
                    "rm_team_name": str(bu.rm_team_name),
                    "opponent_team_id": int(bu.opponent_team_id),
                    "opponent_team_name": str(bu.opponent_team_name),
                    "gk_id": int(bu.gk_id),
                    "gk_name": str(bu.gk_name),
                    "gk_side": str(bu.gk_side),
                    "event_id": str(bu.event_id),
                    "event_reference_time": str(bu.event_reference_time),
                    "setup_start_time": str(bu.setup_start_time),
                    "setup_end_time": str(bu.setup_end_time),
                    "ready_time": str(bu.ready_time),
                    "kick_time": str(bu.kick_time),
                    "time_before_event_s": float(bu.time_before_event_s),
                    "window_start": str(window_start),
                    "window_end": str(window_end),
                    "frames_path": rel_path,
                    "n_rows": int(len(df_slice)),
                }
            )
            diag["build_up_id"] = int(build_up_id)
            diag["frames_path"] = rel_path
            diagnostics_rows.append(diag)

    # Save index
    index_df = pd.DataFrame(index_rows).sort_values(["build_up_id"], ignore_index=True)
    ensure_dir(out_dir)
    index_df.to_parquet(out_dir / "index.parquet", index=False)
    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(
        ["game_id", "period", "event_reference_time", "event_id"], ignore_index=True
    )
    diagnostics_df.to_parquet(out_dir / "goal_kick_diagnostics.parquet", index=False)

    duplicate_rows: list[dict[str, Any]] = []
    dup_df = diagnostics_df[diagnostics_df["failure_reason"] == "duplicate_kick_time"].copy()
    if not dup_df.empty:
        dup_keys = dup_df[["game_id", "period", "selected_kick_time"]].drop_duplicates()
        kept_df = diagnostics_df[diagnostics_df["build_up_detected"] == True].copy()
        kept_for_dup = kept_df.merge(
            dup_keys,
            on=["game_id", "period", "selected_kick_time"],
            how="inner",
        )

        for _, key_row in dup_keys.iterrows():
            game_id = str(key_row["game_id"])
            period = int(key_row["period"])
            kick_time = str(key_row["selected_kick_time"])

            dup_group = dup_df[
                (dup_df["game_id"] == game_id)
                & (dup_df["period"] == period)
                & (dup_df["selected_kick_time"] == kick_time)
            ].copy()
            kept_group = kept_for_dup[
                (kept_for_dup["game_id"] == game_id)
                & (kept_for_dup["period"] == period)
                & (kept_for_dup["selected_kick_time"] == kick_time)
            ].copy()

            merged_refs = pd.concat([kept_group, dup_group], ignore_index=True, sort=False)
            merged_refs = merged_refs.sort_values(["event_reference_time", "event_id"], ignore_index=True)

            event_times = merged_refs["event_reference_time"].astype(str).tolist()
            event_ids = merged_refs["event_id"].astype(str).tolist()
            event_secs = merged_refs["time_before_event_s"].astype(float).tolist()

            first_ref = event_times[0] if event_times else None
            last_ref = event_times[-1] if event_times else None
            min_before = float(min(event_secs)) if event_secs else float("nan")
            max_before = float(max(event_secs)) if event_secs else float("nan")

            duplicate_rows.append(
                {
                    "game_id": game_id,
                    "period": period,
                    "season_name": str(merged_refs["season_name"].dropna().iloc[0]) if "season_name" in merged_refs.columns and merged_refs["season_name"].notna().any() else pd.NA,
                    "season_start_year": (
                        int(merged_refs["season_start_year"].dropna().iloc[0])
                        if "season_start_year" in merged_refs.columns and merged_refs["season_start_year"].notna().any()
                        else pd.NA
                    ),
                    "season_end_year": (
                        int(merged_refs["season_end_year"].dropna().iloc[0])
                        if "season_end_year" in merged_refs.columns and merged_refs["season_end_year"].notna().any()
                        else pd.NA
                    ),
                    "competition_name": str(merged_refs["competition_name"].dropna().iloc[0]) if "competition_name" in merged_refs.columns and merged_refs["competition_name"].notna().any() else pd.NA,
                    "competition_area": str(merged_refs["competition_area"].dropna().iloc[0]) if "competition_area" in merged_refs.columns and merged_refs["competition_area"].notna().any() else pd.NA,
                    "selected_kick_time": kick_time,
                    "selected_kick_frame": int(merged_refs["selected_kick_frame"].dropna().iloc[0]),
                    "n_refs_collapsed": int(len(merged_refs)),
                    "n_duplicate_refs": int(len(dup_group)),
                    "kept_build_up_id": (
                        int(kept_group["build_up_id"].dropna().iloc[0])
                        if not kept_group.empty and kept_group["build_up_id"].notna().any()
                        else pd.NA
                    ),
                    "first_event_reference_time": first_ref,
                    "last_event_reference_time": last_ref,
                    "min_time_before_event_s": min_before,
                    "max_time_before_event_s": max_before,
                    "event_reference_times": " | ".join(event_times),
                    "event_ids": " | ".join(event_ids),
                }
            )

    duplicate_audit_df = pd.DataFrame(
        duplicate_rows,
        columns=[
            "game_id",
            "period",
            "season_name",
            "season_start_year",
            "season_end_year",
            "competition_name",
            "competition_area",
            "selected_kick_time",
            "selected_kick_frame",
            "n_refs_collapsed",
            "n_duplicate_refs",
            "kept_build_up_id",
            "first_event_reference_time",
            "last_event_reference_time",
            "min_time_before_event_s",
            "max_time_before_event_s",
            "event_reference_times",
            "event_ids",
        ],
    )
    duplicate_audit_df = duplicate_audit_df.sort_values(
        ["game_id", "period", "selected_kick_time"], ignore_index=True
    )
    duplicate_audit_df.to_parquet(out_dir / "goal_kick_duplicate_audit.parquet", index=False)

    # Save parameters
    data_root_rel = safe_relpath(data_root, base=APP_ROOT)
    out_dir_rel = safe_relpath(out_dir, base=APP_ROOT)

    params = {
        "created_at": started_at,
        "data_root": data_root_rel or "<custom>",
        "out_dir": out_dir_rel or "<custom>",
        "games": list(game_ids),
        "lookback_seconds": int(lookback_seconds),
        "window_after_seconds": int(window_after_seconds),
        "pre_kick_seconds": float(pre_kick_seconds),
        "start_from_ready": bool(start_from_ready),
        "max_seconds_before_ref": float(max_seconds_before_ref),
        "goal_area_depth_m": float(goal_area_depth_m),
        "goal_area_half_width_m": float(goal_area_half_width_m),
        "goal_area_x_margin_m": float(goal_area_x_margin_m),
        "gk_ball_distance_m": float(gk_ball_distance_m),
        "kick_displacement_m": float(kick_displacement_m),
        "kick_confirm_frames": int(kick_confirm_frames),
        "n_build_ups": int(len(index_rows)),
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    out_dir_display = out_dir_rel or "<custom>"
    return {"out_dir": out_dir_display, "n_build_ups": int(len(index_rows))}


def main() -> None:
    """Main entry point for extraction script."""
    parser = argparse.ArgumentParser(
        description="Offline extractor for RM pressing build-up windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output paths
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Root directory containing raw data (default: data/raw/RealMadrid)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_ROOT,
                        help="Output directory for processed data (default: data/processed/rm_pressing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress information")

    # Game selection (mutually exclusive)
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--full", action="store_true",
                     help="Extract for all games available under data-root")
    sel.add_argument("--match-id", action="append",
                     help="Extract only a specific match id (can be repeated)")

    parser.add_argument("--games", nargs="*", default=None,
                        help="Explicit list of games (defaults to 5-game sample)")
    parser.add_argument("match_ids", nargs="*",
                        help="Optional positional match ids (e.g. `python extraction.py 2014987`)")

    # Tracking data options
    parser.add_argument("--cache-tracking-parquet", action=argparse.BooleanOptionalAction, default=False,
                        help="Cache converted JSON tracking data to parquet")
    parser.add_argument("--rebuild-tracking-parquet", action="store_true",
                        help="Re-convert tracking JSON into tracking_parquet even if parquet exists")

    # Time window parameters
    parser.add_argument("--lookback-seconds", type=int, default=int(DEFAULT_LOOKBACK_SECONDS),
                        help=f"Seconds to look back from goal kick event (default: {DEFAULT_LOOKBACK_SECONDS})")
    parser.add_argument("--window-after-seconds", type=int, default=int(DEFAULT_WINDOW_AFTER_SECONDS),
                        help=f"Seconds to include after kick/ready time (default: {DEFAULT_WINDOW_AFTER_SECONDS})")
    parser.add_argument("--pre-kick-seconds", type=float, default=float(DEFAULT_PRE_KICK_SECONDS),
                        help=f"Seconds before kick to include for kick-centered windows (default: {DEFAULT_PRE_KICK_SECONDS})")
    parser.add_argument("--start-from-ready", action=argparse.BooleanOptionalAction, default=bool(DEFAULT_START_FROM_READY),
                        help=f"Start window from ready time instead of kick time (default: {DEFAULT_START_FROM_READY})")
    parser.add_argument("--max-seconds-before-ref", type=float, default=float(DEFAULT_MAX_SECONDS_BEFORE_REF),
                        help=f"Maximum seconds before event reference to accept (default: {DEFAULT_MAX_SECONDS_BEFORE_REF})")

    # Geometry parameters
    parser.add_argument("--goal-area-depth-m", type=float, default=float(DEFAULT_GOAL_AREA_DEPTH_M),
                        help=f"Goal area depth from goal line in meters (default: {DEFAULT_GOAL_AREA_DEPTH_M})")
    parser.add_argument("--goal-area-half-width-m", type=float, default=float(DEFAULT_GOAL_AREA_HALF_WIDTH_M),
                        help=f"Goal area half-width in meters (default: {DEFAULT_GOAL_AREA_HALF_WIDTH_M})")
    parser.add_argument("--goal-area-x-margin-m", type=float, default=float(DEFAULT_GOAL_AREA_X_MARGIN_M),
                        help=f"Extra margin for goal area in meters (default: {DEFAULT_GOAL_AREA_X_MARGIN_M})")
    parser.add_argument("--gk-ball-distance-m", type=float, default=float(DEFAULT_GK_BALL_DISTANCE_M),
                        help=f"Max GK-ball distance for setup in meters (default: {DEFAULT_GK_BALL_DISTANCE_M})")

    # Kick detection parameters
    parser.add_argument("--kick-displacement-m", type=float, default=float(DEFAULT_KICK_DISPLACEMENT_M),
                        help=f"Min displacement to detect kick in meters (default: {DEFAULT_KICK_DISPLACEMENT_M})")
    parser.add_argument("--kick-confirm-frames", type=int, default=int(DEFAULT_KICK_CONFIRM_FRAMES),
                        help=f"Frames to confirm kick (default: {DEFAULT_KICK_CONFIRM_FRAMES})")

    args = parser.parse_args()

    # Resolve paths
    data_root = resolve_user_path(Path(args.data_root))
    out_dir = resolve_user_path(Path(args.out_dir))

    # Determine which games to process
    if args.full:
        game_ids = list_full_game_ids(data_root)
    elif args.match_id:
        game_ids = [str(g) for g in args.match_id]
    elif args.match_ids:
        game_ids = [str(g) for g in args.match_ids]
    elif args.games:
        game_ids = [str(g) for g in args.games]
    else:
        game_ids = list(DEFAULT_FIVE_GAMES)

    # Run extraction
    result = run_extractor_to_disk(
        data_root=data_root,
        out_dir=out_dir,
        game_ids=game_ids,
        lookback_seconds=int(args.lookback_seconds),
        window_after_seconds=int(args.window_after_seconds),
        pre_kick_seconds=float(args.pre_kick_seconds),
        start_from_ready=bool(args.start_from_ready),
        max_seconds_before_ref=float(args.max_seconds_before_ref),
        goal_area_depth_m=float(args.goal_area_depth_m),
        goal_area_half_width_m=float(args.goal_area_half_width_m),
        goal_area_x_margin_m=float(args.goal_area_x_margin_m),
        gk_ball_distance_m=float(args.gk_ball_distance_m),
        kick_displacement_m=float(args.kick_displacement_m),
        kick_confirm_frames=int(args.kick_confirm_frames),
        cache_tracking_parquet=bool(args.cache_tracking_parquet),
        rebuild_tracking_parquet=bool(args.rebuild_tracking_parquet),
        verbose=bool(args.verbose),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
