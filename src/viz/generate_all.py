import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tqdm import tqdm
import json
import glob
from src.viz.cluster_comparison import plot_cluster_comparison
from src.viz.individual_player import plot_individual_player_movement
from src.viz.pressing_heatmap import (
    calculate_pressing_event_density,
    plot_heatmap
)
from src.utils.pickle_compat import load_pickle_compat_or_raise

from src.viz.plots import plot_cluster_summary, plot_topic_heatmap, draw_pitch
from src.features.services.window_loader import WindowLoader
from src.features.services.normalization import normalize_coordinates
from src.features.services.utils import prepare_frame_data, time_to_seconds
from src.features.services.metadata import enrich_with_team_id
from src.features.services.possession import infer_ball_carrier
from src.models.gmm_zones import identify_pressers, GMMConfig

# --- VELOCITY CALCULATION HELPER FUNCTIONS ---
def get_frame_at_time(df, target_t):
    """
    Get frame closest to target time.
    Returns: DataFrame slice at closest timestamp, or None if not found.
    """
    try:
        time_diff = (df['time_seconds'] - target_t).abs()
        closest_idx = time_diff.idxmin()
        if time_diff[closest_idx] > 2.0:  # More than 2 seconds away
            return None
        # Get closest timestamp
        closest_ts = df.loc[closest_idx, 'time_seconds']
        return df[df['time_seconds'] == closest_ts]
    except:
        return None


def get_ball_position_from_frame(frame):
    """
    Return the ball position from a prepared/normalized tracking frame.

    prepare_frame_data() merges ball coordinates onto player rows, so most
    visualization frames no longer contain standalone is_ball rows.
    """
    if frame is None or frame.empty:
        return None

    for x_col, y_col in (("ball_x_norm", "ball_y_norm"), ("ball_x", "ball_y")):
        if {x_col, y_col}.issubset(frame.columns):
            coords = frame[[x_col, y_col]].dropna()
            if coords.empty:
                continue
            coords = coords.apply(pd.to_numeric, errors="coerce").dropna()
            if not coords.empty:
                return np.array([coords[x_col].mean(), coords[y_col].mean()], dtype=float)

    if "is_ball" in frame.columns:
        ball_rows = frame[frame["is_ball"] == True]
        for x_col, y_col in (("x_norm", "y_norm"), ("x", "y")):
            if not ball_rows.empty and {x_col, y_col}.issubset(ball_rows.columns):
                coords = ball_rows[[x_col, y_col]].dropna()
                if coords.empty:
                    continue
                coords = coords.apply(pd.to_numeric, errors="coerce").dropna()
                if not coords.empty:
                    return np.array([coords[x_col].mean(), coords[y_col].mean()], dtype=float)

    return None


def get_interpolated_positions(df, target_t, team_id, window_size=1.0, is_ball=False):
    """
    Get player positions at target time with temporal interpolation.

    Uses a time window around target_t to find player positions, interpolating
    from nearby frames when players are missing from the exact target frame.

    Args:
        df: DataFrame with tracking data
        target_t: Target time in seconds
        team_id: Team ID to filter (or None for all teams)
        window_size: Time window in seconds (default 1.0 = ±1s around target)
        is_ball: If True, get ball position instead of players

    Returns:
        dict: {player_id: np.array([x, y])} or tuple (x, y) for ball
    """
    try:
        # Get frames within the time window
        window_df = df[(df['time_seconds'] >= target_t - window_size) &
                       (df['time_seconds'] <= target_t + window_size)]

        if window_df.empty:
            return {} if not is_ball else None

        if is_ball:
            return get_ball_position_from_frame(window_df)
        else:
            # Get player positions
            if team_id is not None:
                player_df = window_df[(window_df['team_id'] == team_id) & (window_df['is_ball'] == False)]
            else:
                player_df = window_df[window_df['is_ball'] == False]

            if player_df.empty:
                return {}

            # For each player, get their average position in the window
            # Prefer frames closer to target_t
            positions = {}
            for pid in player_df['player_id'].unique():
                player_frames = player_df[player_df['player_id'] == pid]
                # Weight by proximity to target time
                time_weights = 1.0 / (1.0 + np.abs(player_frames['time_seconds'] - target_t))
                weighted_x = (player_frames['x_norm'] * time_weights).sum() / time_weights.sum()
                weighted_y = (player_frames['y_norm'] * time_weights).sum() / time_weights.sum()
                positions[int(pid)] = np.array([weighted_x, weighted_y])

            return positions
    except Exception as e:
        return {} if not is_ball else None


def get_build_up_anchor_time(df_norm, loader, build_up_id):
    """
    Resolve the temporal anchor for visualization.

    Prefer the detected kick time from build-up metadata so cluster plots are
    aligned to the actual goal-kick event rather than the start of the stored
    tracking window.
    """
    t_start = float(df_norm["time_seconds"].min())
    t_end = float(df_norm["time_seconds"].max())

    try:
        meta = loader.get_metadata(int(build_up_id))
        kick_time = meta.get("kick_time")
        if kick_time is None or pd.isna(kick_time):
            return t_start

        return float(time_to_seconds(str(kick_time)))
    except Exception:
        return t_start


def is_in_left_penalty_area(x, y, margin_m=0.0):
    return (-52.5 - margin_m) <= float(x) <= (-36.0 + margin_m) and abs(float(y)) <= (20.15 + margin_m)


def project_outside_left_penalty(pos, eps_m=0.5):
    """Keep aggregated restart dots from visually violating the goal-kick law."""
    x, y = float(pos[0]), float(pos[1])
    if not is_in_left_penalty_area(x, y):
        return np.array([x, y])

    candidates = [
        (abs(-36.0 - x), np.array([-36.0 + eps_m, y])),
        (abs(20.15 - y), np.array([x, 20.15 + eps_m])),
        (abs(-20.15 - y), np.array([x, -20.15 - eps_m])),
    ]
    return min(candidates, key=lambda item: item[0])[1]


def constrain_rm_restart_positions(pos_by_pid):
    return {
        pid: project_outside_left_penalty(pos)
        for pid, pos in pos_by_pid.items()
    }


def is_in_left_goal_area(x, y, margin_m=2.5):
    return (-52.5 - margin_m) <= float(x) <= (-47.0 + margin_m) and abs(float(y)) <= (9.15 + margin_m)


def rm_players_in_left_penalty(frame, opp_id, margin_m=0.0):
    rm_rows = frame[(frame["team_id"] != opp_id) & (frame["is_ball"] == False) & frame["team_id"].notna()]
    if rm_rows.empty:
        return 0
    return int(
        sum(
            is_in_left_penalty_area(row["x_norm"], row["y_norm"], margin_m=margin_m)
            for _, row in rm_rows.iterrows()
        )
    )


def ball_in_left_goal_area_frame(frame, margin_m=2.5):
    ball = get_ball_position_from_frame(frame)
    if ball is None:
        return False
    return is_in_left_goal_area(ball[0], ball[1], margin_m=margin_m)


def get_legal_restart_frame(df_norm, kick_t, opp_id, lookback_s=3.0):
    """
    Find the latest legal goal-kick restart scene before ball movement.

    The detected kick time is the first ball-movement frame. For the law-critical
    visualization panel, use the latest preceding frame where the ball is still
    in the goal area and RM players are outside the penalty area.
    """
    try:
        window = df_norm[
            (df_norm["time_seconds"] >= float(kick_t) - float(lookback_s))
            & (df_norm["time_seconds"] <= float(kick_t))
        ]
        if window.empty:
            return None

        goal_area_fallback = None
        for frame_time in sorted(window["time_seconds"].dropna().unique(), reverse=True):
            frame = window[window["time_seconds"] == frame_time]
            if ball_in_left_goal_area_frame(frame):
                if goal_area_fallback is None:
                    goal_area_fallback = frame
                if rm_players_in_left_penalty(frame, opp_id) == 0:
                    return frame
        if goal_area_fallback is not None:
            return goal_area_fallback
    except Exception:
        return None
    return None


def make_safe_folder_name(value):
    """Create a stable folder label for season names such as 2023/2024."""
    text = str(value or "UnknownSeason")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "UnknownSeason"


def clear_cluster_visualization_artifacts(sub_dir):
    """Remove generated cluster plots so stale local-GMM files do not linger."""
    generated_patterns = [
        "cluster_comparison_*.png",
        "cluster_sequence_*.png",
        "individual_trigger_*.png",
        "individual_support*.png",
        "individual_blocker_*.png",
    ]
    for pattern in generated_patterns:
        for path in sub_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def is_goalkeeper_player(pid, player_info, pos_map):
    """Return True when player metadata identifies the player as a goalkeeper."""
    info = player_info.get(int(pid), {})
    role = str(info.get("role", ""))
    if "Goalkeeper" in role:
        return True

    if int(pid) in pos_map:
        group, name = pos_map[int(pid)]
        group = str(group)
        name = str(name)
        if group == "Goalkeeper" or "Goalkeeper" in name:
            return True
    return False


def select_rm_display_pids(player_counts, player_info, pos_map, total_players=11):
    """
    Select one RM goalkeeper plus the most represented outfield players.

    Cluster sequence/comparison plots describe the full defensive shape, so
    they should show 11 RM players when the data contains them.
    """
    if not player_counts:
        return set()

    ranked = sorted(player_counts, key=lambda pid: player_counts[pid], reverse=True)
    goalkeepers = [pid for pid in ranked if is_goalkeeper_player(pid, player_info, pos_map)]
    outfield = [pid for pid in ranked if not is_goalkeeper_player(pid, player_info, pos_map)]

    selected = []
    if goalkeepers:
        selected.append(goalkeepers[0])

    selected.extend(outfield[: max(total_players - len(selected), 0)])
    return set(selected[:total_players])


def format_subset_name(subset_name):
    """
    Format subset name for display.
    'ShortRestarts' -> 'Short Restarts'
    'DirectRestarts' -> 'Direct Restarts'
    """
    if subset_name == "ShortRestarts":
        return "Short Restarts"
    elif subset_name == "DirectRestarts":
        return "Direct Restarts"
    else:
        return subset_name


def calculate_player_velocity_between_frames(df_norm, player_id, t_start, t_end, team_id_filter=None):
    """
    Calculate velocity vector for player between two timepoints.

    Args:
        df_norm: Normalized tracking dataframe with time_seconds column
        player_id: Player ID to track
        t_start: Start time in seconds
        t_end: End time in seconds
        team_id_filter: Optional team ID to filter by

    Returns:
        velocity: (vx, vy) in m/s, or None if data missing
    """
    frame_start = get_frame_at_time(df_norm, t_start)
    frame_end = get_frame_at_time(df_norm, t_end)

    if frame_start is None or frame_end is None:
        return None

    player_start = frame_start[frame_start['player_id'] == player_id]
    player_end = frame_end[frame_end['player_id'] == player_id]

    if player_start.empty or player_end.empty:
        return None

    pos_start = np.array([player_start.iloc[0]['x_norm'], player_start.iloc[0]['y_norm']])
    pos_end = np.array([player_end.iloc[0]['x_norm'], player_end.iloc[0]['y_norm']])

    dt = t_end - t_start
    if dt == 0:
        return None

    velocity = (pos_end - pos_start) / dt

    return velocity

def calculate_velocity_towards_target(player_pos, player_vel, target_pos):
    """
    Calculate component of velocity in direction of target.

    Args:
        player_pos: (x, y) position of player
        player_vel: (vx, vy) velocity vector, or None
        target_pos: (x, y) position of target (e.g., ball)

    Returns:
        scalar velocity towards target (m/s), 0 if moving away or velocity is None
    """
    if player_vel is None:
        return 0.0

    to_target = target_pos - player_pos
    distance = np.linalg.norm(to_target)

    if distance < 0.1:  # Already at target
        return 0.0

    direction = to_target / distance
    vel_towards = np.dot(player_vel, direction)

    return max(0.0, vel_towards)  # Only count approaching movement

def load_overlay_data(loader: WindowLoader, valid_ids: list, gk_side_map: dict, game_id_map: dict, opp_id_map: dict):
    rm_points = []
    opp_points = []
    player_points = {} 
    
    for bid in tqdm(valid_ids, desc="Loading Overlay Data"):
        try:
            df = loader.load_build_up(bid)
            df = prepare_frame_data(df)
            gid = game_id_map.get(bid, 0)
            df = enrich_with_team_id(df, gid)
            side = gk_side_map.get(bid, "left")
            df_norm = normalize_coordinates(df, side)
            
            opp_id = opp_id_map.get(bid)
            
            opp_mask = df_norm["team_id"] == opp_id
            if opp_mask.any():
                opp_pts = df_norm.loc[opp_mask, ["x_norm", "y_norm"]].values
                opp_points.append(opp_pts)
                
            rm_mask = (df_norm["team_id"] != opp_id) & (~df_norm["is_ball"]) & (df_norm["team_id"].notna())
            if rm_mask.any():
                rm_pts = df_norm.loc[rm_mask, ["x_norm", "y_norm"]].values
                rm_points.append(rm_pts)
                for pid, px, py in df_norm.loc[rm_mask, ["player_id", "x_norm", "y_norm"]].values:
                    try:
                        pid = int(pid)
                    except:
                        continue # Skip invalid PIDs
                        
                    if pid not in player_points:
                        player_points[pid] = []
                    player_points[pid].append([px, py])
                    
        except Exception as e:
            continue
            
    rm_all = np.vstack(rm_points) if rm_points else np.empty((0, 2))
    opp_all = np.vstack(opp_points) if opp_points else np.empty((0, 2))
    
    for pid in player_points:
        player_points[pid] = np.array(player_points[pid])
        
    return rm_all, opp_all, player_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="data/processed/rm_pressing_features/features.parquet")
    parser.add_argument("--topics-dir", default="data/processed/rm_pressing_topics")
    parser.add_argument("--out-dir", default="visualizations")
    parser.add_argument("--processed-root", default="data/processed/rm_pressing")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics_dir = Path(args.topics_dir)
    
    print("Loading features and models...")
    features_df = pd.read_parquet(args.features_path)
    
    loader = WindowLoader(args.processed_root)
    meta_df = loader.index
    season_cols = [
        "build_up_id",
        "season_name",
        "season_start_year",
        "season_end_year",
        "competition_name",
        "competition_area",
    ]
    available_season_cols = [c for c in season_cols if c in meta_df.columns]
    if "build_up_id" in available_season_cols:
        features_df = features_df.merge(
            meta_df[available_season_cols].drop_duplicates("build_up_id"),
            on="build_up_id",
            how="left",
        )
    if "season_name" not in features_df.columns:
        features_df["season_name"] = "AllSeasons"
    if "restart_type" not in features_df.columns and "gk_kick_distance_m" in features_df.columns:
        features_df["restart_type"] = np.where(
            features_df["gk_kick_distance_m"] < 15.0,
            "short",
            "direct",
        )
        features_df.loc[features_df["gk_kick_distance_m"].isna(), "restart_type"] = "unknown"
    if "restart_distance_band" not in features_df.columns and "gk_kick_distance_m" in features_df.columns:
        features_df["restart_distance_band"] = np.select(
            [
                features_df["gk_kick_distance_m"] < 15.0,
                features_df["gk_kick_distance_m"] < 30.0,
                features_df["gk_kick_distance_m"] >= 30.0,
            ],
            ["short_under_15m", "medium_15_30m", "long_30m_plus"],
            default="unknown",
        )
    if "is_true_long_restart" not in features_df.columns and "gk_kick_distance_m" in features_df.columns:
        features_df["is_true_long_restart"] = features_df["gk_kick_distance_m"] >= 30.0
    gk_side_map = dict(zip(meta_df["build_up_id"], meta_df["gk_side"]))
    game_id_map = dict(zip(meta_df["build_up_id"], meta_df["game_id"]))
    opp_id_map = dict(zip(meta_df["build_up_id"], meta_df["opponent_team_id"]))
    
    gmm_init = load_pickle_compat_or_raise(topics_dir / "gmm_initial.pkl")
    gmm_target = load_pickle_compat_or_raise(topics_dir / "gmm_target.pkl")
    clusters_path = topics_dir / "clusters.parquet"
    nmf_cluster_map = None
    if clusters_path.exists():
        clusters_df = pd.read_parquet(clusters_path)
        if {"build_up_id", "cluster_id"}.issubset(clusters_df.columns):
            nmf_cluster_map = {
                int(row["build_up_id"]): int(row["cluster_id"])
                for _, row in clusters_df.dropna(subset=["build_up_id", "cluster_id"]).iterrows()
            }
            n_model_clusters = clusters_df["cluster_id"].nunique()
            print(f"Loaded NMF cluster labels: {n_model_clusters} clusters, {len(nmf_cluster_map)} build-ups")
        else:
            print(f"Warning: {clusters_path} is missing build_up_id/cluster_id columns.")
    else:
        print(f"Warning: {clusters_path} not found. Run src.models.clustering before cluster visualizations.")
    
    # Load Player Metadata
    print("Loading Player Metadata...")
    player_info = {}
    raw_meta_dir = Path("data/raw/RealMadrid/meta")
    
    if raw_meta_dir.exists():
        json_files = glob.glob(str(raw_meta_dir / "*.json"))
        for jf in tqdm(json_files, desc="Parsing Meta"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "players" in data:
                    for p in data["players"]:
                        try:
                            # Ensure PID is hashable (int)
                            try:
                                pid = int(p["id"])
                            except:
                                continue 
                                
                            name = p.get("short_name")
                            if not name:
                                name = p.get("last_name")
                            if not name:
                                name = str(pid)
                                
                            role = p.get("player_role", {}).get("name", "Unknown")
                            team_id = p.get("team_id")
                            number = p.get("number", 0) # Default to 0 if missing
                            
                            existing = player_info.get(pid)
                            if existing is None:
                                player_info[pid] = {
                                    "name": name,
                                    "role": role,
                                    "team_id": team_id,
                                    "number": number,
                                }
                            elif "Goalkeeper" in str(role) and "Goalkeeper" not in str(existing.get("role", "")):
                                # The same player can appear as a substitute in one match and as GK in another.
                                # Keep the goalkeeper role when any match metadata confirms it.
                                existing["role"] = role
                                existing["number"] = number
                        except Exception as inner_e:
                            # print(f"Error parsing player: {inner_e}")
                            continue
            except:
                continue
    else:
        print("Warning: Raw metadata not found. Using IDs.")
        
    # --- RESTORE RM METADATA ---
    rm_names = {}
    rm_numbers = {}
    all_rm_player_pids = []
    for pid, info in player_info.items():
        # 262 is Real Madrid (Found via debug)
        if info.get("team_id") == 262:
             rm_names[pid] = info["name"]
             rm_numbers[pid] = info["number"]
             all_rm_player_pids.append(pid)
    
    # Fallback if 217 not found (e.g. using different ID in this dataset?)
    if not all_rm_player_pids:
        print("Warning: No RM players found with Team ID 217. Checking player_info...")
        # Optional: Print first few to debug
        # print(list(player_info.values())[:3])

    try:
        pos_df = pd.read_parquet("data/processed/metadata/players.parquet")
        pos_map = {
            int(row["player_id"]): (row["position_group"], row["position_name"])
            for _, row in pos_df.iterrows()
        }
    except Exception:
        pos_map = {}

    subsets = {}
    restart_type_filters = {
        "ShortRestarts": features_df["restart_type"] == "short",
        "DirectRestarts": features_df["restart_type"] == "direct",
    }
    for base_name, mask in restart_type_filters.items():
        subset_df = features_df[mask].copy()
        for season_name, season_df in subset_df.groupby("season_name", dropna=False):
            season_label = str(season_name or "Unknown Season")
            season_folder = make_safe_folder_name(season_label)
            subset_key = f"{base_name}/{season_folder}"
            subsets[subset_key] = {
                "base_name": base_name,
                "season_label": season_label,
                "season_folder": season_folder,
                "ids": season_df["build_up_id"].tolist(),
            }
    
    # ---------------------------
    # GLOBAL SCALING PRE-PASS
    # Collect node sizes and edge weights across all subsets for cross-plot comparability
    # ---------------------------
    print("Computing global scaling parameters for cross-plot comparability...")
    global_node_sizes = []
    global_edge_weights = []
    
    # Import required for normalization
    from src.viz.network import PressureNetworkPlotter
    
    # Process each subset to collect scaling statistics
    for name, subset_info in subsets.items():
        ids = subset_info["ids"]
        if not ids:
            continue
            
        presser_positions_temp = {}
        co_press_matrix_temp = {}
        pid_press_counts_temp = {}
        gmm_config = GMMConfig()
        
        for bid in tqdm(ids, desc=f"Collecting {name} data for scaling"):
            try:
                df = loader.load_build_up(bid)
                df = prepare_frame_data(df)
                gid = game_id_map.get(bid, 0)
                df = enrich_with_team_id(df, gid)
                side = gk_side_map.get(bid, "left")
                df_norm = normalize_coordinates(df, side)
                
                opp_id = opp_id_map.get(bid)
                df_norm = infer_ball_carrier(df_norm, opp_id)
                
                if "time_seconds" not in df_norm.columns:
                    df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

                current_pressers = identify_pressers(df_norm, gmm_config)
                
                if not current_pressers:
                    continue
                
                filtered_pressers = []
                for pid in current_pressers:
                    try:
                        pid_int = int(pid)
                    except:
                        continue
                        
                    role = player_info.get(pid_int, {}).get("role", "")
                    if "Goalkeeper" in role:
                        continue
                        
                    filtered_pressers.append(pid_int)
                    pid_press_counts_temp[pid_int] = pid_press_counts_temp.get(pid_int, 0) + 1
                        
                filtered_pressers = sorted(list(set(filtered_pressers)))
                for i in range(len(filtered_pressers)):
                    for j in range(i+1, len(filtered_pressers)):
                        p1, p2 = filtered_pressers[i], filtered_pressers[j]
                        key = (p1, p2)
                        co_press_matrix_temp[key] = co_press_matrix_temp.get(key, 0) + 1
                        
            except Exception as e:
                continue
        
        # Collect node sizes and edge weights from this subset
        global_node_sizes.extend(pid_press_counts_temp.values())
        global_edge_weights.extend(co_press_matrix_temp.values())
    
    # Compute global scaling parameters
    if global_node_sizes:
        global_node_sizes_array = np.array(global_node_sizes, dtype=float)
        global_node_scale = PressureNetworkPlotter._robust_norm(global_node_sizes_array, 5.0, 95.0)
    else:
        global_node_scale = None  # Fall back to local scaling
    
    if global_edge_weights:
        global_edge_weights_array = np.array(global_edge_weights, dtype=float)
        _, global_edge_scale = PressureNetworkPlotter._robust_norm(global_edge_weights_array, 5.0, 95.0)
        global_edge_scale = max(global_edge_scale, 1.0)
    else:
        global_edge_scale = None  # Fall back to local scaling
    
    print(f"Global node scale: {global_node_scale}")
    print(f"Global edge scale: {global_edge_scale}")
    
    # ---------------------------
    # MAIN PROCESSING LOOP
    # ---------------------------
    
    for name, subset_info in subsets.items():
        ids = subset_info["ids"]
        base_name = subset_info["base_name"]
        season_label = subset_info["season_label"]
        season_folder = subset_info["season_folder"]
        if not ids:
            print(f"Subset {name} empty, skipping.")
            continue
            
        print(f"Processing Subset: {base_name} / {season_label} ({len(ids)} items)")
        sub_dir = out_dir / base_name / season_folder
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        rm_pts, opp_pts, player_pts = load_overlay_data(loader, ids, gk_side_map, game_id_map, opp_id_map)
        
        # 1. GMM Zones (Disabled)
        # 2. Player Analysis (Disabled)
            
        # 3. Pressure Network
        print(f"Generating Pressure Network for {base_name} / {season_label}...")
        
        presser_positions = {} 
        co_press_matrix = {} 
        pid_press_counts = {} 
        
        gmm_config = GMMConfig()
        
        for bid in tqdm(ids, desc="Network Data"):
            try:
                df = loader.load_build_up(bid)
                df = prepare_frame_data(df)
                gid = game_id_map.get(bid, 0)
                df = enrich_with_team_id(df, gid)
                side = gk_side_map.get(bid, "left")
                df_norm = normalize_coordinates(df, side)
                
                opp_id = opp_id_map.get(bid)
                df_norm = infer_ball_carrier(df_norm, opp_id)
                
                if "time_seconds" not in df_norm.columns:
                    df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

                current_pressers = identify_pressers(df_norm, gmm_config)
                
                if not current_pressers:
                    continue
                
                filtered_pressers = []
                for pid in current_pressers:
                    # Cast to int for lookup
                    try:
                        pid_int = int(pid)
                    except:
                        continue # Skip invalid PIDs that can't be cast to int
                        
                    # Filter GK
                    role = player_info.get(pid_int, {}).get("role", "")
                    if "Goalkeeper" in role:
                        continue
                        
                    # Use pid_int for consistency
                    filtered_pressers.append(pid_int)
                    
                    # We accept that df matches float or int based on equality
                    p_rows = df_norm[df_norm["player_id"] == pid]
                   
                    if not p_rows.empty:
                        avg_x = p_rows["x_norm"].mean()
                        avg_y = p_rows["y_norm"].mean()
                        
                        if pid_int not in presser_positions:
                            presser_positions[pid_int] = []
                        presser_positions[pid_int].append([avg_x, avg_y])
                        
                        pid_press_counts[pid_int] = pid_press_counts.get(pid_int, 0) + 1
                        
                filtered_pressers = sorted(list(set(filtered_pressers)))
                for i in range(len(filtered_pressers)):
                    for j in range(i+1, len(filtered_pressers)):
                        p1, p2 = filtered_pressers[i], filtered_pressers[j]
                        key = (p1, p2)
                        co_press_matrix[key] = co_press_matrix.get(key, 0) + 1
                        
            except Exception as e:
                continue

        # Formation-Based Selection (User Request: Strict 4-3-3 with Sides & CB Split)
        # Load Position Map if not loaded
        if "pos_map" not in locals():
            try:
                pos_df = pd.read_parquet("data/processed/metadata/players.parquet")
                # Create map: pid -> (group, name)
                pos_map = {}
                for _, row in pos_df.iterrows():
                    pos_map[row['player_id']] = (row['position_group'], row['position_name'])
            except:
                pos_map = {}

        # Sort all candidates by frequency
        all_pids = list(pid_press_counts.keys())
        all_pids.sort(key=lambda p: pid_press_counts[p], reverse=True)
        
        # Target Slots: 4-3-3 (Strict Sides + CB Split)
        # We try to fill specific sides first.
        # LB, LCB, RCB, RB
        # LW, CF, RW
        # MID x 3
        
        slots = {
            "LB": 1, "RB": 1, 
            "LCB": 1, "RCB": 1, # Specific CBs
            "MID": 3, 
            "LW": 1, "RW": 1, "CF": 1
        }
        
        selected_pids = set()
        
        # Helper to classify granular roles
        def get_granular_role(pid):
            # Check player_info first for goalkeeper detection
            p_meta = player_info.get(pid, {})
            role_str = p_meta.get("role", "")
            if "Goalkeeper" in role_str:
                return "GK"

            if pid not in pos_map: return "UNK"
            group, name = pos_map[pid]

            # Goalkeeper check from pos_map
            if "Goalkeeper" in name or group == "Goalkeeper":
                return "GK"

            # Defenders
            if "Left Back" in name or "Left Wing Back" in name: return "LB"
            if "Right Back" in name or "Right Wing Back" in name: return "RB"

            # CBs
            if "Left Center Back" in name: return "LCB"
            if "Right Center Back" in name: return "RCB"
             # Fallback for generic CBs handled later or via fuzzy match logic?
            if group == "Central Defender": return "CB" # Generic

            # Attackers (Wingers)
            if "Left Winger" in name or "Left Forward" in name: return "LW"
            if "Right Winger" in name or "Right Forward" in name or "Right Midfield" in name: return "RW"
            if "Left Midfield" in name: return "LW"

            # Central
            if group == "Center Forward": return "CF"
            if group == "Midfield": return "MID"

            if group == "Full Back": return "FB"
            if group == "Wide Attacker": return "WINGER"

            if group == "Other" and "Substitute" in name: return "UNK"
            return "UNK"

        # 1. Fill specific slots
        for pid in all_pids:
            role = get_granular_role(pid)
            if role in slots and slots[role] > 0:
                selected_pids.add(pid)
                slots[role] -= 1
        
        # 2. handle Generic Fallbacks if slots missing
        # If LCB or RCB missing, use generic "CB"
        # If LB or RB missing, use generic "FB"
        
        needed_cb = slots["LCB"] + slots["RCB"]
        needed_fb = slots["LB"] + slots["RB"]
        
        for pid in all_pids:
            if pid in selected_pids: continue
            
            role = get_granular_role(pid)
            if role == "CB" and needed_cb > 0:
                selected_pids.add(pid)
                needed_cb -= 1
            elif role == "FB" and needed_fb > 0:
                selected_pids.add(pid)
                needed_fb -= 1
                
        # 3. Fill remaining to reach 10 OUTFIELD players (Wildcards)
        target_count = 10  # Changed from 11 to exclude goalkeeper
        for pid in all_pids:
            if len(selected_pids) >= target_count:
                break
            if pid not in selected_pids:
                # Exclude GK
                role = player_info.get(pid, {}).get("role", "")
                if "Goalkeeper" in role:
                    continue
                if pid in pos_map and pos_map[pid][1] == "Goalkeeper":
                    continue
                # Additional check using granular role function
                if get_granular_role(pid) == "GK":
                    continue

                selected_pids.add(pid)

        top_11_pids = selected_pids
        
        # Prepare Data for Plot (Include ALL outfield players, not just Top 11)
        final_avg_pos = {}
        final_names = {}
        
        for pid, pts in presser_positions.items():
            # Include everyone who is not GK (already filtered)
            try:
                pts = np.array(pts)
                final_avg_pos[pid] = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
                name_val = player_info.get(pid, {}).get("name", str(pid))
                final_names[pid] = name_val
            except Exception as e:
                continue
            
        final_edges = co_press_matrix # Pass all edges
            
        # Prepare Numbers & Roles Map
        final_numbers = {}
        final_roles = {}
        
        # Define Mapping (Mirrors pressing_network.py)
        ROLE_CATEGORY_MAP = {
            "CF": "FW", "ST": "FW", "RW": "FW", "LW": "FW", "RF": "FW", "LF": "FW",
            "CM": "MF", "DM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
            "LDM": "MF", "RDM": "MF", "LAM": "MF", "RAM": "MF",
            "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF",
            "LCB": "DF", "RCB": "DF", "CD": "DF",
            "GK": "GK"
        }

        for pid in final_names:
            final_numbers[pid] = player_info.get(pid, {}).get("number", 0)
            
            # Extract Role
            raw_role = player_info.get(pid, {}).get("role", "Unknown")
            # Or from pos_map if available
            if raw_role == "Unknown" and pid in pos_map:
                # Try to map from group
                grp = pos_map[pid][0]
                if grp == "Center Forward" or grp == "Wide Attacker": raw_role = "CF" # Proxy
                elif grp == "Midfield": raw_role = "CM"
                elif grp == "Defenders" or grp == "Central Defender" or grp == "Full Back": raw_role = "CB"
            
            final_roles[pid] = ROLE_CATEGORY_MAP.get(raw_role, "UNK")

        # Format subset name for display
        formatted_name = f"{format_subset_name(base_name)} - {season_label}"

        plot_title = f"{formatted_name} - Pressure Network"

        if final_avg_pos:
            from src.viz.network import PressureNetworkPlotter, PressureNetworkStyle
            
            # Instantiate Plotter with default style
            plotter = PressureNetworkPlotter(PressureNetworkStyle())

            # Version 1: Full Team (with grey bubbles)
            plotter.plot(
                avg_positions=final_avg_pos, 
                co_press_counts=final_edges, 
                node_sizes=pid_press_counts, 
                player_names=final_names,
                player_numbers=final_numbers,
                player_roles=final_roles,
                highlight_pids=top_11_pids, 
                title=plot_title,
                out_path=str(sub_dir / "pressure_network_fullteam.png"),
                show_all_players=True,
                node_scale=global_node_scale,
                edge_scale=global_edge_scale
            )
            
            # Version 2: Frequent Players Only (no grey bubbles)
            plotter.plot(
                avg_positions=final_avg_pos,
                co_press_counts=final_edges,
                node_sizes=pid_press_counts,
                player_names=final_names,
                player_numbers=final_numbers,
                player_roles=final_roles,
                highlight_pids=top_11_pids,
                title=plot_title,
                out_path=str(sub_dir / "pressure_network_frequentplayers.png"),
                show_all_players=False,
                node_scale=global_node_scale,
                edge_scale=global_edge_scale
            )

            # Version 3 & 4: Affinity-Weighted Networks
            print(f"Generating Affinity-Weighted Networks for {formatted_name}...")
            from src.models.pressing_affinity import PressingAffinityCalculator

            affinity_scores = PressingAffinityCalculator.calculate_affinity_matrix(
                co_press_counts=final_edges,
                node_sizes=pid_press_counts,
                method="jaccard"
            )

            # Version 3: Full Team with Affinity Weights
            plotter.plot(
                avg_positions=final_avg_pos,
                co_press_counts=final_edges,
                node_sizes=pid_press_counts,
                player_names=final_names,
                player_numbers=final_numbers,
                player_roles=final_roles,
                highlight_pids=top_11_pids,
                title=f"{plot_title} (Affinity)",
                out_path=str(sub_dir / "pressure_network_fullteam_affinity.png"),
                show_all_players=True,
                node_scale=global_node_scale,
                edge_scale=global_edge_scale,
                use_affinity_weights=True,
                affinity_scores=affinity_scores
            )

            # Version 4: Frequent Players with Affinity Weights
            plotter.plot(
                avg_positions=final_avg_pos,
                co_press_counts=final_edges,
                node_sizes=pid_press_counts,
                player_names=final_names,
                player_numbers=final_numbers,
                player_roles=final_roles,
                highlight_pids=top_11_pids,
                title=f"{plot_title} (Affinity)",
                out_path=str(sub_dir / "pressure_network_frequentplayers_affinity.png"),
                show_all_players=False,
                node_scale=global_node_scale,
                edge_scale=global_edge_scale,
                use_affinity_weights=True,
                affinity_scores=affinity_scores
            )

            # Version 5: Centrality Analysis
            print(f"Generating Centrality Analysis for {formatted_name}...")
            from src.analysis.network_centrality import PressingNetworkAnalyzer

            analyzer = PressingNetworkAnalyzer(
                avg_positions=final_avg_pos,
                co_press_counts=final_edges,
                node_sizes=pid_press_counts
            )

            # Export centrality metrics to CSV
            analyzer.export_centrality_report(
                player_names=final_names,
                output_path=str(sub_dir / "centrality_metrics.csv")
            )

            # Identify and print key players
            key_players = analyzer.identify_key_players()
            # Use encode with errors='replace' to handle Unicode in Windows console
            try:
                print(f"  Pressing Orchestrators (Betweenness): {[(final_names.get(pid, pid), f'{score:.3f}') for pid, score in key_players['orchestrators']]}")
                print(f"  Pressing Hubs (Degree): {[(final_names.get(pid, pid), f'{score:.3f}') for pid, score in key_players['hubs']]}")
                print(f"  Tight Units (Clustering): {[(final_names.get(pid, pid), f'{score:.3f}') for pid, score in key_players['tight_units']]}")
            except UnicodeEncodeError:
                print(f"  Key players identified (Unicode display error)")
                pass

            # Calculate network metrics
            network_metrics = analyzer.calculate_network_metrics()
            print(f"  Network Density: {network_metrics.density:.3f}")
            print(f"  Avg Clustering: {network_metrics.avg_clustering:.3f}")
            print(f"  Communities Detected: {network_metrics.num_communities}")

            # Generate centrality visualization
            centrality_metrics = analyzer.calculate_centrality()
            plotter.plot_with_centrality(
                avg_positions=final_avg_pos,
                co_press_counts=final_edges,
                node_sizes=pid_press_counts,
                centrality_metrics=centrality_metrics,
                player_names=final_names,
                player_numbers=final_numbers,
                player_roles=final_roles,
                highlight_pids=top_11_pids,
                title=f"{plot_title} - Pressing Orchestrators & Units",
                out_path=str(sub_dir / "pressure_network_centrality.png"),
                show_all_players=False,
                node_scale=global_node_scale
            )

            # Version 6 & 7: Temporal Phase Networks
            print(f"Generating Temporal Phase Networks for {formatted_name}...")
            from src.viz.pressing_network import TemporalPressingAggregator

            temporal_agg = TemporalPressingAggregator()

            # Collect pressing data with temporal information
            for bid in tqdm(ids, desc="Temporal Phase Data"):
                try:
                    df = loader.load_build_up(bid)
                    df = prepare_frame_data(df)
                    gid = game_id_map.get(bid, 0)
                    df = enrich_with_team_id(df, gid)
                    side = gk_side_map.get(bid, "left")
                    df_norm = normalize_coordinates(df, side)

                    opp_id = opp_id_map.get(bid)
                    df_norm = infer_ball_carrier(df_norm, opp_id)

                    if "time_seconds" not in df_norm.columns:
                        df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

                    # Calculate build-up duration
                    duration = df_norm["time_seconds"].max() - df_norm["time_seconds"].min()

                    # Collect pressing frames with timestamps
                    pressing_frames = []
                    for frame_time, frame_group in df_norm.groupby("time_seconds"):
                        current_pressers = identify_pressers(frame_group, gmm_config)

                        if not current_pressers:
                            continue

                        # Filter out goalkeepers
                        filtered_pressers = []
                        for pid in current_pressers:
                            try:
                                pid_int = int(pid)
                            except:
                                continue

                            role = player_info.get(pid_int, {}).get("role", "")
                            if "Goalkeeper" in role:
                                continue

                            p_rows = frame_group[frame_group["player_id"] == pid]
                            if not p_rows.empty:
                                filtered_pressers.append({
                                    'player_id': pid_int,
                                    'x': p_rows["x_norm"].mean(),
                                    'y': p_rows["y_norm"].mean()
                                })

                        if filtered_pressers:
                            # Normalize timestamp to start at 0
                            normalized_time = frame_time - df_norm["time_seconds"].min()
                            pressing_frames.append((normalized_time, filtered_pressers))

                    if pressing_frames and duration > 0:
                        temporal_agg.add_build_up(pressing_frames, duration)

                except Exception:
                    continue

            # Get statistics
            stats = temporal_agg.get_statistics()
            print(f"  Total build-ups: {stats['total_build_ups']}")
            print(f"  Avg duration: {stats['avg_duration']:.1f}s")
            print(f"  Early frames: {stats['early_frames']}, Late frames: {stats['late_frames']}")

            # Generate early phase network
            early_positions, early_edges = temporal_agg.get_phase_data("early")
            if early_positions:
                plotter.plot(
                    avg_positions=early_positions,
                    co_press_counts=early_edges,
                    node_sizes=pid_press_counts,  # Use overall counts for consistency
                    player_names=final_names,
                    player_numbers=final_numbers,
                    player_roles=final_roles,
                    highlight_pids=top_11_pids,
                    title=f"{plot_title} - Early Pressing (First 33%)",
                    out_path=str(sub_dir / "pressure_network_early_phase.png"),
                    show_all_players=False,
                    node_scale=global_node_scale
                )

            # Generate late phase network
            late_positions, late_edges = temporal_agg.get_phase_data("late")
            if late_positions:
                plotter.plot(
                    avg_positions=late_positions,
                    co_press_counts=late_edges,
                    node_sizes=pid_press_counts,  # Use overall counts for consistency
                    player_names=final_names,
                    player_numbers=final_numbers,
                    player_roles=final_roles,
                    highlight_pids=top_11_pids,
                    title=f"{plot_title} - Late Pressing (Last 33%)",
                    out_path=str(sub_dir / "pressure_network_late_phase.png"),
                    show_all_players=False,
                    node_scale=global_node_scale
                )



        # 4. Cluster Comparison Overlays (Skip for "All")
        if base_name != "All":
            print(f"Generating NMF Cluster Comparisons for {formatted_name}...")
            from src.viz.cluster_comparison import plot_cluster_comparison
            from collections import defaultdict

            if nmf_cluster_map is None:
                print("NMF cluster labels unavailable, skipping cluster visualizations.")
                continue
            clear_cluster_visualization_artifacts(sub_dir)

            # Use the saved NMF/Agglomerative cluster labels so visualizations
            # match the validated topic/clustering model exactly.
            cluster_data = defaultdict(list)  # cluster_id -> list of build_up_ids
            missing_cluster_labels = 0
            for bid in ids:
                cluster_id = nmf_cluster_map.get(int(bid))
                if cluster_id is None:
                    missing_cluster_labels += 1
                    continue
                cluster_data[cluster_id].append(int(bid))
            
            cluster_counts = {cid: len(bids) for cid, bids in cluster_data.items()}
            top_clusters = sorted(cluster_counts.items(), key=lambda x: x[0])
            
            if missing_cluster_labels:
                print(f"Warning: {missing_cluster_labels} build-ups in {formatted_name} had no NMF cluster label.")
            if not top_clusters:
                print(f"No labelled NMF clusters found for {formatted_name}, skipping.")
                continue
            print(f"NMF clusters for {formatted_name}: {[(c, cnt) for c, cnt in top_clusters]}")
            
            # Generate comparison for each top cluster
            for cluster_id, count in top_clusters:
                cluster_bids = cluster_data[cluster_id]
                plot_title = f"NMF Cluster {cluster_id} - {formatted_name} (N={count})"
                out_path = sub_dir / f"cluster_comparison_{cluster_id}.png"
                
                # ... [Existing Aggregation Logic: initial_rm, initial_opp, avg_ball_init ...] ...
                # reconstruct the aggregation block here because I am replacing it
                # To be partial: I need to locate where `avg_ball_init` and `final_initial_rm` are available.
                # They are computed inside the loop. I need to insert logic AFTER they are computed.
                
                # --- RE-COMPUTING AGGREGATION FOR CLUSTER ---
                init_rm_agg = defaultdict(list)
                init_opp_agg = defaultdict(list)
                init_ball_agg = []
                
                targ_rm_agg = defaultdict(list)
                targ_opp_agg = defaultdict(list)
                targ_ball_agg = []
                
                # New: T+2 Aggregation for Momentum
                init_rm_t2_agg = defaultdict(list)
                init_ball_t2_agg = []
                
                for bid in cluster_bids:
                    try:
                        df = loader.load_build_up(bid)
                        df = prepare_frame_data(df)
                        gid = game_id_map.get(bid, 0)
                        df = enrich_with_team_id(df, gid)
                        side = gk_side_map.get(bid, "left")
                        df_norm = normalize_coordinates(df, side)
                        opp_id = opp_id_map.get(bid)
                        df_norm = infer_ball_carrier(df_norm, opp_id)
                        if "time_seconds" not in df_norm.columns:
                            df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

                        t_start = get_build_up_anchor_time(df_norm, loader, bid)
                        rm_team_id = df_norm[df_norm['team_id'] != opp_id]['team_id'].iloc[0] if len(df_norm[df_norm['team_id'] != opp_id]) > 0 else None

                        # Initial State: use the latest legal restart frame before ball movement.
                        restart_frame = get_legal_restart_frame(df_norm, t_start, opp_id)
                        if restart_frame is None or restart_frame.empty:
                            restart_frame = get_frame_at_time(df_norm, t_start)
                        if restart_frame is not None and not restart_frame.empty:
                            init_opp_rows = restart_frame[restart_frame["team_id"] == opp_id]
                            for _, row in init_opp_rows.iterrows():
                                init_opp_agg[int(row["player_id"])].append([row["x_norm"], row["y_norm"]])

                            init_rm_rows = restart_frame[
                                (restart_frame["team_id"] == rm_team_id)
                                & (restart_frame["is_ball"] == False)
                            ]
                            for _, row in init_rm_rows.iterrows():
                                pid = int(row["player_id"])
                                if pid in all_rm_player_pids:
                                    init_rm_agg[pid].append([row["x_norm"], row["y_norm"]])

                            ball_pos = get_ball_position_from_frame(restart_frame)
                            if ball_pos is not None:
                                init_ball_agg.append(ball_pos)

                        # Momentum State (t=2) - Use interpolation with 1s window
                        t2_rm_pos = get_interpolated_positions(df_norm, t_start + 2.0, rm_team_id, window_size=1.0)
                        for pid, pos in t2_rm_pos.items():
                            if pid in all_rm_player_pids:
                                init_rm_t2_agg[pid].append(pos)

                        t2_ball_pos = get_interpolated_positions(df_norm, t_start + 2.0, None, window_size=1.0, is_ball=True)
                        if t2_ball_pos is not None:
                            init_ball_t2_agg.append(t2_ball_pos)

                        # Target State (t=10) - Use interpolation with 1s window
                        targ_opp_pos = get_interpolated_positions(df_norm, t_start + 10.0, opp_id, window_size=1.0)
                        for pid, pos in targ_opp_pos.items():
                            targ_opp_agg[pid].append(pos)

                        targ_rm_pos = get_interpolated_positions(df_norm, t_start + 10.0, rm_team_id, window_size=1.0)
                        for pid, pos in targ_rm_pos.items():
                            if pid in all_rm_player_pids:
                                targ_rm_agg[pid].append(pos)

                        targ_ball_pos = get_interpolated_positions(df_norm, t_start + 10.0, None, window_size=1.0, is_ball=True)
                        if targ_ball_pos is not None:
                            targ_ball_agg.append(targ_ball_pos)
                    except: continue

                # Finalize Averages
                # Use consistent "top N most frequent" approach for BOTH teams
                # This ensures stable player counts across clusters for better comparisons

                min_threshold = max(count * 0.15, 2)  # At least 15% or 2 sequences

                # RM Players - full display XI: goalkeeper + 10 outfield players.
                rm_display_counts = {
                    pid: len(pts)
                    for pid, pts in init_rm_agg.items()
                    if len(pts) >= min_threshold
                }
                display_rm_pids = select_rm_display_pids(
                    rm_display_counts,
                    player_info,
                    pos_map,
                    total_players=11,
                )
                final_initial_rm = {
                    pid: np.mean(init_rm_agg[pid], axis=0)
                    for pid in display_rm_pids
                    if pid in init_rm_agg
                }
                final_initial_rm = constrain_rm_restart_positions(final_initial_rm)
                final_rm_t2 = {
                    pid: np.mean(init_rm_t2_agg[pid], axis=0)
                    for pid in display_rm_pids
                    if pid in init_rm_t2_agg
                }
                final_target_rm = {
                    pid: np.mean(targ_rm_agg[pid], axis=0)
                    for pid in display_rm_pids
                    if pid in targ_rm_agg
                }

                # Opposition Players - Initial state (top 11 players)
                init_opp_candidates = [(pid, np.mean(pts, axis=0), len(pts)) for pid, pts in init_opp_agg.items() if len(pts) >= min_threshold]
                init_opp_candidates.sort(key=lambda x: x[2], reverse=True)
                final_initial_opp = {pid: pos for pid, pos, _ in init_opp_candidates[:11]}

                # Opposition Players - Target state (top 11 players)
                targ_opp_candidates = [(pid, np.mean(pts, axis=0), len(pts)) for pid, pts in targ_opp_agg.items() if len(pts) >= min_threshold]
                targ_opp_candidates.sort(key=lambda x: x[2], reverse=True)
                final_target_opp = {pid: pos for pid, pos, _ in targ_opp_candidates[:11]}

                avg_ball_init = np.mean(init_ball_agg, axis=0) if init_ball_agg else (0,0)
                avg_ball_t2 = np.mean(init_ball_t2_agg, axis=0) if init_ball_t2_agg else (0,0)
                avg_ball_targ = np.mean(targ_ball_agg, axis=0) if targ_ball_agg else (0,0)

                # Debug: Print player counts
                print(f"  Cluster {cluster_id}: RM={len(final_initial_rm)} players, OPP Initial={len(final_initial_opp)}, OPP Target={len(final_target_opp)}")

                # --- IMPROVED TRIGGER & SUPPORT IDENTIFICATION ---
                # Calculate weighted scores: velocity towards ball + distance closed

                rm_scores = []

                for pid, pos_t0 in final_initial_rm.items():
                    # Skip if player not present at t=2
                    if pid not in final_rm_t2:
                        continue

                    pos_t2 = final_rm_t2[pid]

                    # Estimate velocity at t=0→t=1 using positions at t=0 and t=2
                    # Assume approximately linear motion
                    estimated_velocity = (pos_t2 - pos_t0) / 2.0  # Average velocity over 2 seconds

                    # Calculate velocity component towards ball
                    vel_towards_ball = calculate_velocity_towards_target(
                        pos_t0, estimated_velocity, avg_ball_init
                    )

                    # Distance to ball at t=0
                    dist_t0 = np.linalg.norm(pos_t0 - avg_ball_init)

                    # Distance closed (t=0 to t=2)
                    dist_t2 = np.linalg.norm(pos_t2 - avg_ball_t2)
                    distance_closed = dist_t0 - dist_t2  # Positive = getting closer

                    # Weighted score (prioritize velocity: 60%, distance closed: 40%)
                    score = (vel_towards_ball * 0.6) + (distance_closed * 0.4)

                    rm_scores.append((score, dist_t0, pid))

                # Sort by score (descending)
                rm_scores.sort(reverse=True, key=lambda x: x[0])

                # Trigger = highest score
                trigger_pid = rm_scores[0][2] if len(rm_scores) > 0 else None

                # Support = next 2 best players after trigger (always take 2 if available)
                if len(rm_scores) >= 3:
                    support_pids = [rm_scores[1][2], rm_scores[2][2]]
                elif len(rm_scores) == 2:
                    support_pids = [rm_scores[1][2]]
                else:
                    support_pids = []

                # Debug: Print trigger identification info
                try:
                    if len(rm_scores) > 0:
                        top_score, top_dist, top_pid = rm_scores[0]
                        top_name = rm_names.get(top_pid, f"Player {top_pid}")
                        print(f"  Trigger: {top_name} (score={top_score:.2f}, dist={top_dist:.1f}m)")
                    if len(support_pids) > 0:
                        support_names = [rm_names.get(pid, f"Player {pid}") for pid in support_pids]
                        support_scores = [f"{score:.2f}" for score, dist, pid in rm_scores[1:] if pid in support_pids]
                        print(f"  Support: {', '.join(support_names)} (scores: {', '.join(support_scores)})")
                except UnicodeEncodeError:
                    print(f"  Trigger/Support players identified (Unicode display error)")
                    pass

                # 2. Channel Blocker (Target State: Closest to Ball, EXCLUDING Pressers)
                rm_targ_dists = []
                excluded_pids = {trigger_pid} | set(support_pids)
                
                for pid, pos in final_target_rm.items():
                    if pid in excluded_pids: continue
                    d = np.linalg.norm(pos - avg_ball_targ)
                    rm_targ_dists.append((d, pid))
                rm_targ_dists.sort()
                
                blocker_pid = rm_targ_dists[0][1] if len(rm_targ_dists) > 0 else None

                cluster_title = f"NMF Cluster {cluster_id} - {formatted_name} (N={count})"
                print(f"Plotting Cluster {cluster_id} Comparison...")
                plot_args = {
                    'initial_opp': final_initial_opp,
                    'initial_rm': final_initial_rm,
                    'initial_ball': avg_ball_init,
                    'target_opp': final_target_opp,
                    'target_rm': final_target_rm,
                    'target_ball': avg_ball_targ,
                    'rm_player_names': rm_names,
                    'rm_player_numbers': rm_numbers,
                    'title': cluster_title,
                    'out_path': out_path,
                    'trigger_pid': trigger_pid,
                    'support_pids': support_pids,
                    'blocker_pid': blocker_pid
                }
                plot_cluster_comparison(**plot_args)
                
                # --- TEMPORAL SEQUENCE ---
                print(f"Cluster {cluster_id}: Generating Temporal Sequence (kick, +2,+4,+6,+8,+10)")
                from src.viz.sequence import plot_temporal_sequence
                # ... [Sequence Logic Continues] ...
                
                # Extract opponent and RM positions for this cluster
                # Prepare data containers for separated visualization
                initial_opp_pos = defaultdict(list)
                target_opp_pos = defaultdict(list)
                
                initial_rm_pos = defaultdict(list)
                target_rm_pos = defaultdict(list)
                rm_pos_t2 = defaultdict(list)  # Add t=2 for trigger calculation

                initial_ball_pos = []
                target_ball_pos = []
                ball_pos_t2 = []  # Add t=2 for trigger calculation
                
                rm_cluster_press_counts = defaultdict(int) 
                rm_cluster_copresses = defaultdict(int)
                rm_presence_counts = defaultdict(int)
                
                for bid in cluster_bids:
                    try:
                        df = loader.load_build_up(bid)
                        df = prepare_frame_data(df)
                        gid = game_id_map.get(bid, 0)
                        df = enrich_with_team_id(df, gid)
                        side = gk_side_map.get(bid, "left")
                        df_norm = normalize_coordinates(df, side)
                        
                        opp_id = opp_id_map.get(bid)
                        df_norm = infer_ball_carrier(df_norm, opp_id)
                        
                        if "time_seconds" not in df_norm.columns:
                            df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
                        
                        # Define Timestamps
                        t_start = get_build_up_anchor_time(df_norm, loader, bid)
                        t_2 = t_start + 2.0  # t=2 for trigger calculation
                        t_target = t_start + 10.0

                        # Helper to get frame at time t
                        def get_frame_at_time(df, target_t):
                            try:
                                time_diff = (df['time_seconds'] - target_t).abs()
                                closest_idx = time_diff.idxmin()
                                if time_diff[closest_idx] > 2.0: return None
                                # Get closest timestamp
                                closest_ts = df.loc[closest_idx, 'time_seconds']
                                return df[df['time_seconds'] == closest_ts]
                            except:
                                return None

                        frame_initial = get_legal_restart_frame(df_norm, t_start, opp_id)
                        frame_t2 = get_frame_at_time(df_norm, t_2)
                        frame_target = get_frame_at_time(df_norm, t_target)
                        
                        # Fallback for short sequences
                        if frame_target is None or frame_target.empty:
                             t_end = df_norm["time_seconds"].max()
                             if (t_end - t_start) > 5.0:
                                frame_target = get_frame_at_time(df_norm, t_end)
                        
                        rm_mask_query = f"team_id != {opp_id} and is_ball == False"
                        opp_mask_query = f"team_id == {opp_id}"
                        # Track RM presence for selection
                        current_rm = df_norm.query(rm_mask_query)["player_id"].unique()
                        for pid in current_rm:
                            try:
                                rm_presence_counts[int(pid)] += 1
                                rm_cluster_press_counts[int(pid)] += 1 
                            except: continue

                        # COLLECT INITIAL DATA
                        if frame_initial is not None and not frame_initial.empty:
                            # Opponent
                            opp_rows = frame_initial.query(opp_mask_query)
                            for _, row in opp_rows.iterrows():
                                try:
                                    initial_opp_pos[int(row['player_id'])].append([row['x_norm'], row['y_norm']])
                                except: continue
                            
                            # RM
                            rm_rows = frame_initial.query(rm_mask_query)
                            for _, row in rm_rows.iterrows():
                                try:
                                    initial_rm_pos[int(row['player_id'])].append([row['x_norm'], row['y_norm']])
                                except: continue
                                
                            # Ball
                            ball_pos = get_ball_position_from_frame(frame_initial)
                            if ball_pos is not None:
                                initial_ball_pos.append(ball_pos)

                        # COLLECT TARGET DATA
                        if frame_target is not None and not frame_target.empty:
                            # Opponent
                            opp_rows = frame_target.query(opp_mask_query)
                            for _, row in opp_rows.iterrows():
                                try:
                                    target_opp_pos[int(row['player_id'])].append([row['x_norm'], row['y_norm']])
                                except: continue

                            # RM
                            rm_rows = frame_target.query(rm_mask_query)
                            for _, row in rm_rows.iterrows():
                                try:
                                    target_rm_pos[int(row['player_id'])].append([row['x_norm'], row['y_norm']])
                                except: continue

                            # Ball
                            ball_pos = get_ball_position_from_frame(frame_target)
                            if ball_pos is not None:
                                target_ball_pos.append(ball_pos)

                        # COLLECT t=2 DATA (for trigger calculation)
                        if frame_t2 is not None and not frame_t2.empty:
                            # RM at t=2
                            rm_rows_t2 = frame_t2.query(rm_mask_query)
                            for _, row in rm_rows_t2.iterrows():
                                try:
                                    rm_pos_t2[int(row['player_id'])].append([row['x_norm'], row['y_norm']])
                                except: continue

                            # Ball at t=2
                            ball_pos = get_ball_position_from_frame(frame_t2)
                            if ball_pos is not None:
                                ball_pos_t2.append(ball_pos)

                    except Exception as e:
                        continue
                # --- Average Positions ---
                def get_avg_pos(pos_dict):
                    avg = {}
                    for pid, coords in pos_dict.items():
                        if len(coords) > 0:
                            avg[pid] = np.mean(coords, axis=0)
                    return avg
                
                avg_initial_opp = get_avg_pos(initial_opp_pos)
                avg_target_opp = get_avg_pos(target_opp_pos)
                
                avg_initial_rm = get_avg_pos(initial_rm_pos)
                avg_target_rm = get_avg_pos(target_rm_pos)
                
                # Filter Opponent (Top 11)
                opp_counts = {p: len(coords) for p, coords in initial_opp_pos.items()} 
                top_opp = sorted(opp_counts.keys(), key=lambda x: opp_counts[x], reverse=True)[:11]
                
                final_initial_opp = {p: avg_initial_opp[p] for p in top_opp if p in avg_initial_opp}
                final_target_opp = {p: avg_target_opp[p] for p in top_opp if p in avg_target_opp} 
                
                print(f"Cluster {cluster_id}: RM press counts: {dict(list(rm_cluster_press_counts.items())[:5])}")
                
                rm_display_counts = {
                    pid: len(coords)
                    for pid, coords in initial_rm_pos.items()
                    if len(coords) >= min_threshold
                }
                if len(rm_display_counts) < 11:
                    rm_display_counts = dict(rm_cluster_press_counts)

                selected_rm_pids = select_rm_display_pids(
                    rm_display_counts,
                    player_info,
                    pos_map,
                    total_players=11,
                )
                
                print(f"Cluster {cluster_id}: Selected {len(selected_rm_pids)} RM players for highlighting")
                print(f"Cluster {cluster_id}: Selected PIDs: {list(selected_rm_pids)[:5]}")
                
                # The global RM name/number maps already include both seasons.
                
                # NOTE: Duplicate cluster comparison generation removed
                # The cluster comparison is already generated above (line ~1089) with trigger/support/blocker
                # This section was creating a duplicate visualization with the same filename, overwriting the first

                # print(f"Cluster {cluster_id}: Generating Side-by-Side comparison")
                # # Generate Side-by-Side Comparison
                # cluster_title = f"Cluster {cluster_id} Comparison - {formatted_name}"
                # out_path = sub_dir / f"cluster_comparison_{cluster_id}.png"
                #
                # # Filter RM positions for selected players
                # final_initial_rm = {p: avg_initial_rm[p] for p in selected_rm_pids if p in avg_initial_rm}
                # final_target_rm = {p: avg_target_rm[p] for p in selected_rm_pids if p in avg_target_rm}
                #
                # # Ball Averages
                # avg_ball_init = tuple(np.mean(initial_ball_pos, axis=0)) if initial_ball_pos else None
                # avg_ball_targ = tuple(np.mean(target_ball_pos, axis=0)) if target_ball_pos else None
                # avg_ball_t2 = np.mean(ball_pos_t2, axis=0) if ball_pos_t2 else None
                #
                # # --- IDENTIFY TRIGGER, SUPPORT, AND BLOCKER PLAYERS ---
                # # Average t=2 RM positions
                # avg_rm_t2 = {}
                # for pid, coords in rm_pos_t2.items():
                #     if len(coords) > 0:
                #         avg_rm_t2[pid] = np.mean(coords, axis=0)
                #
                # # Convert to numpy arrays for calculations
                # final_initial_rm_np = {pid: np.array(pos) for pid, pos in final_initial_rm.items()}
                # final_rm_t2 = {pid: avg_rm_t2[pid] for pid in final_initial_rm if pid in avg_rm_t2}
                #
                # # 1. Trigger Player (Highest pressing score at t=0)
                # trigger_pid = None
                # support_pids = []
                # blocker_pid = None
                #
                # if avg_ball_init is not None and avg_ball_t2 is not None:
                #     avg_ball_init_np = np.array(avg_ball_init)
                #     avg_ball_t2_np = np.array(avg_ball_t2)
                #
                #     rm_scores = []
                #     for pid, pos_t0 in final_initial_rm_np.items():
                #         if pid not in final_rm_t2:
                #             continue
                #
                #         pos_t2 = final_rm_t2[pid]
                #
                #         # Estimate velocity at t=0→t=1 using positions at t=0 and t=2
                #         estimated_velocity = (pos_t2 - pos_t0) / 2.0
                #
                #         # Calculate velocity component towards ball
                #         vel_towards_ball = calculate_velocity_towards_target(
                #             pos_t0, estimated_velocity, avg_ball_init_np
                #         )
                #
                #         # Distance to ball at t=0
                #         dist_t0 = np.linalg.norm(pos_t0 - avg_ball_init_np)
                #
                #         # Distance closed (t=0 to t=2)
                #         dist_t2 = np.linalg.norm(pos_t2 - avg_ball_t2_np)
                #         distance_closed = dist_t0 - dist_t2
                #
                #         # Weighted score
                #         score = (vel_towards_ball * 0.6) + (distance_closed * 0.4)
                #         rm_scores.append((score, dist_t0, pid))
                #
                #     # Sort by score
                #     rm_scores.sort(reverse=True, key=lambda x: x[0])
                #
                #     # Trigger = highest score
                #     trigger_pid = rm_scores[0][2] if len(rm_scores) > 0 else None
                #
                #     # Support = next 2 best players
                #     if len(rm_scores) >= 3:
                #         support_pids = [rm_scores[1][2], rm_scores[2][2]]
                #     elif len(rm_scores) == 2:
                #         support_pids = [rm_scores[1][2]]
                #
                # # 2. Channel Blocker (Target State: Closest to Ball, EXCLUDING Pressers)
                # if avg_ball_targ is not None:
                #     avg_ball_targ_np = np.array(avg_ball_targ)
                #     rm_targ_dists = []
                #     excluded_pids = {trigger_pid} | set(support_pids)
                #
                #     for pid, pos in final_target_rm.items():
                #         if pid in excluded_pids:
                #             continue
                #         pos_np = np.array(pos)
                #         d = np.linalg.norm(pos_np - avg_ball_targ_np)
                #         rm_targ_dists.append((d, pid))
                #
                #     rm_targ_dists.sort()
                #     blocker_pid = rm_targ_dists[0][1] if len(rm_targ_dists) > 0 else None
                #
                # plot_cluster_comparison(
                #     initial_opp=final_initial_opp,
                #     initial_rm=final_initial_rm,
                #     initial_ball=avg_ball_init,
                #     target_opp=final_target_opp,
                #     target_rm=final_target_rm,
                #     target_ball=avg_ball_targ,
                #     rm_player_names=rm_names,
                #     rm_player_numbers=rm_numbers,
                #     title=cluster_title,
                #     out_path=out_path,
                #     trigger_pid=trigger_pid,
                #     support_pids=support_pids,
                #     blocker_pid=blocker_pid
                # )
                
                # --- TEMPORAL SEQUENCE (User Request: Grid 2x2 for t=2,4,6,8,10) ---
                print(f"Cluster {cluster_id}: Generating Temporal Sequence (kick, +2,+4,+6,+8,+10)")
                from src.viz.sequence import plot_temporal_sequence
                
                deltas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
                temporal_data_raw = defaultdict(
                    lambda: {
                        'opp': defaultdict(list),
                        'rm': defaultdict(list),
                        'ball': [],
                        'build_ups': set(),
                    }
                )
                
                # Select a stable opponent XI from post-kick frames. Never use an
                # empty selection as "all players"; that can mix substitutes into
                # the average and distort the blue shape.
                opp_sequence_counts = defaultdict(int)
                for bid in cluster_bids:
                    try:
                        df = loader.load_build_up(bid)
                        df = prepare_frame_data(df)
                        gid = game_id_map.get(bid, 0)
                        df = enrich_with_team_id(df, gid)
                        side = gk_side_map.get(bid, "left")
                        df_norm = normalize_coordinates(df, side)
                        opp_id = opp_id_map.get(bid)
                        if "time_seconds" not in df_norm.columns:
                            df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)

                        t_start = get_build_up_anchor_time(df_norm, loader, bid)
                        for dt in [2.0, 4.0, 6.0, 8.0, 10.0]:
                            frame = get_frame_at_time(df_norm, t_start + dt)
                            if frame is None or frame.empty:
                                continue
                            for pid in frame.loc[frame["team_id"] == opp_id, "player_id"].dropna().unique():
                                opp_sequence_counts[int(pid)] += 1
                    except Exception:
                        continue

                top_opp_pids = {
                    pid
                    for pid, _ in sorted(
                        opp_sequence_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:11]
                }
                if not top_opp_pids:
                    top_opp_pids = set(list(final_target_opp.keys())[:11] or list(final_initial_opp.keys())[:11])
                
                # Collect Data
                for bid in cluster_bids:
                    try:
                        df = loader.load_build_up(bid)
                        df = prepare_frame_data(df)
                        gid = game_id_map.get(bid, 0)
                        df = enrich_with_team_id(df, gid)
                        side = gk_side_map.get(bid, "left")
                        df_norm = normalize_coordinates(df, side)
                        opp_id = opp_id_map.get(bid)
                        df_norm = infer_ball_carrier(df_norm, opp_id)
                        if "time_seconds" not in df_norm.columns:
                            df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds)
                            
                        # Align temporal snapshots to the detected kick time.
                        t_start = get_build_up_anchor_time(df_norm, loader, bid)
                        
                        for dt in deltas:
                            target_t = t_start + dt
                            if dt == 0.0:
                                frame = get_legal_restart_frame(df_norm, t_start, opp_id)
                                if frame is None or frame.empty:
                                    frame = get_frame_at_time(df_norm, target_t)
                            else:
                                # Reuse get_frame_at_time (defined in scope above)
                                frame = get_frame_at_time(df_norm, target_t)
                            
                            if frame is not None and not frame.empty:
                                temporal_data_raw[dt]['build_ups'].add(int(bid))

                                # Opp (Strict Filter: Only Top 11)
                                opp_rows = frame.query(f"team_id == {opp_id}")
                                for _, row in opp_rows.iterrows():
                                    pid = int(row['player_id'])
                                    if pid in top_opp_pids:
                                        temporal_data_raw[dt]['opp'][pid].append([row['x_norm'], row['y_norm']])
                                    
                                # RM (Strict Filter: Only Selected 11)
                                rm_rows = frame.query(f"team_id != {opp_id} and is_ball == False")
                                for _, row in rm_rows.iterrows():
                                    pid = int(row['player_id'])
                                    if pid in selected_rm_pids:
                                        temporal_data_raw[dt]['rm'][pid].append([row['x_norm'], row['y_norm']])
                                        
                                # Ball
                                ball_pos = get_ball_position_from_frame(frame)
                                if ball_pos is not None:
                                    temporal_data_raw[dt]['ball'].append(ball_pos)

                    except: continue

                # Average Data
                temporal_data_final = {}
                for dt in deltas:
                    dt_data = temporal_data_raw[dt]
                    final_frame = {
                        'rm': {},
                        'opp': {},
                        'ball': None,
                        'n_build_ups': len(dt_data['build_ups']),
                        'total_build_ups': int(count),
                    }
                    
                    # Avg RM
                    for pid, pts in dt_data['rm'].items():
                        if pts: final_frame['rm'][pid] = np.mean(pts, axis=0)
                    if dt == 0.0:
                        final_frame['rm'] = constrain_rm_restart_positions(final_frame['rm'])
                            
                    # Avg Opp
                    for pid, pts in dt_data['opp'].items():
                        if pts: final_frame['opp'][pid] = np.mean(pts, axis=0)
                             
                    # Avg Ball
                    if dt_data['ball']:
                        final_frame['ball'] = np.mean(dt_data['ball'], axis=0)
                        
                    temporal_data_final[dt] = final_frame
                    
                # Plot
                seq_title = f"NMF Cluster {cluster_id} Sequence - {formatted_name} (N={count})"
                seq_out_path = sub_dir / f"cluster_sequence_{cluster_id}.png"
                
                plot_temporal_sequence(
                    temporal_data_final,
                    rm_names,
                    rm_numbers,
                    seq_title,
                    seq_out_path,
                    trigger_pid=trigger_pid,
                    support_pids=support_pids,
                    blocker_pid=blocker_pid
                )

                # --- INDIVIDUAL PLAYER MOVEMENT VISUALIZATIONS ---
                print(f"Cluster {cluster_id}: Generating individual player movement visualizations")

                # Prepare temporal data for individual players (include t=0)
                # We need t=0, 2, 4, 6, 8, 10
                individual_timestamps = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

                # Extend temporal_data_final to include t=0
                temporal_with_t0 = {0.0: {'rm': final_initial_rm, 'opp': final_initial_opp, 'ball': avg_ball_init}}
                temporal_with_t0.update(temporal_data_final)

                # Generate visualization for trigger player
                if trigger_pid and trigger_pid in rm_names:
                    trigger_positions = {t: temporal_with_t0[t]['rm'].get(trigger_pid)
                                        for t in individual_timestamps
                                        if t in temporal_with_t0 and trigger_pid in temporal_with_t0[t]['rm']}
                    trigger_ball = {t: temporal_with_t0[t]['ball']
                                   for t in individual_timestamps
                                   if t in temporal_with_t0 and temporal_with_t0[t]['ball'] is not None}
                    trigger_opp = {t: temporal_with_t0[t]['opp']
                                  for t in individual_timestamps
                                  if t in temporal_with_t0}

                    individual_title = f"{formatted_name} - {rm_names[trigger_pid]} (Trigger)"
                    individual_out = sub_dir / f"individual_trigger_{cluster_id}.png"
                    plot_individual_player_movement(
                        trigger_pid,
                        rm_names[trigger_pid],
                        rm_numbers.get(trigger_pid, 0),
                        trigger_positions,
                        trigger_ball,
                        trigger_opp,
                        'trigger',
                        individual_title,
                        individual_out
                    )

                # Generate visualization for support players
                if support_pids:
                    for i, support_pid in enumerate(support_pids):
                        if support_pid in rm_names:
                            support_positions = {t: temporal_with_t0[t]['rm'].get(support_pid)
                                               for t in individual_timestamps
                                               if t in temporal_with_t0 and support_pid in temporal_with_t0[t]['rm']}
                            support_ball = {t: temporal_with_t0[t]['ball']
                                          for t in individual_timestamps
                                          if t in temporal_with_t0 and temporal_with_t0[t]['ball'] is not None}
                            support_opp = {t: temporal_with_t0[t]['opp']
                                         for t in individual_timestamps
                                         if t in temporal_with_t0}

                            individual_title = f"{formatted_name} - {rm_names[support_pid]} (Support)"
                            individual_out = sub_dir / f"individual_support{i+1}_{cluster_id}.png"
                            plot_individual_player_movement(
                                support_pid,
                                rm_names[support_pid],
                                rm_numbers.get(support_pid, 0),
                                support_positions,
                                support_ball,
                                support_opp,
                                'support',
                                individual_title,
                                individual_out
                            )

                # Generate visualization for blocker player
                if blocker_pid and blocker_pid in rm_names:
                    blocker_positions = {t: temporal_with_t0[t]['rm'].get(blocker_pid)
                                        for t in individual_timestamps
                                        if t in temporal_with_t0 and blocker_pid in temporal_with_t0[t]['rm']}
                    blocker_ball = {t: temporal_with_t0[t]['ball']
                                   for t in individual_timestamps
                                   if t in temporal_with_t0 and temporal_with_t0[t]['ball'] is not None}
                    blocker_opp = {t: temporal_with_t0[t]['opp']
                                  for t in individual_timestamps
                                  if t in temporal_with_t0}

                    individual_title = f"{formatted_name} - {rm_names[blocker_pid]} (Blocker)"
                    individual_out = sub_dir / f"individual_blocker_{cluster_id}.png"
                    plot_individual_player_movement(
                        blocker_pid,
                        rm_names[blocker_pid],
                        rm_numbers.get(blocker_pid, 0),
                        blocker_positions,
                        blocker_ball,
                        blocker_opp,
                        'blocker',
                        individual_title,
                        individual_out
                    )

            # --- PRESSING HEATMAPS (GENERATE ONCE PER SUBSET) ---
            print(f"\nGenerating pressing heatmaps for {formatted_name}...")

            # Calculate pressing event density heatmap
            pressing_grid = calculate_pressing_event_density(
                ids,
                loader,
                game_id_map,
                gk_side_map,
                opp_id_map,
                prepare_frame_data,
                normalize_coordinates,
                enrich_with_team_id,
                infer_ball_carrier,
                time_to_seconds
            )

            # Plot pressing initiation heatmap
            pressing_heatmap_title = f"{formatted_name} - Pressing Initiation Heatmap"
            pressing_heatmap_out = sub_dir / f"heatmap_pressing_events_{base_name}_{season_folder}.png"
            plot_heatmap(
                pressing_grid,
                pressing_heatmap_title,
                pressing_heatmap_out,
                "Pressing Initiations per Cell",
                cmap='Reds'
            )

if __name__ == "__main__":
    main()
