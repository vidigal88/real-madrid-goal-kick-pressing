"""
src/viz/pressing_network.py

Tactical Analysis Module for Real Madrid.
- Calculates pressing links per frame.
- Aggregates data over full matches to find 'Average Shape'.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ROLE_CATEGORY_MAP = {
    "CF": "FW", "ST": "FW", "RW": "FW", "LW": "FW",
    "CM": "MF", "DM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
    "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF",
    "GK": "GK"
}

# --- HELPER FUNCTIONS (Per Frame) ---

def get_player_role(info: Dict[str, Any]) -> str:
    """Safely extracts player role (FW, MF, DF)."""
    raw_role = info.get("role")
    return ROLE_CATEGORY_MAP.get(raw_role, "Unknown")

def calculate_pressing_links(
    frame_data: Dict[str, Any],
    player_map: Dict[int, Dict[str, Any]],
    rm_team_id: int,
    ball_carrier_id: Optional[int],
    pressure_threshold_m: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Calculates pressing links for a SINGLE frame.
    """
    if ball_carrier_id is None:
        return []

    # Get Carrier Position
    carrier_pos = None
    for p in frame_data.get("player_data", []):
        if int(p["player_id"]) == int(ball_carrier_id):
            carrier_pos = (float(p["x"]), float(p["y"]))
            break
    
    if not carrier_pos:
        return []

    cx, cy = carrier_pos
    pressing_players = []

    for p in frame_data.get("player_data", []):
        pid = int(p["player_id"])
        info = player_map.get(pid, {})

        if info.get("team_id") != rm_team_id:
            continue

        px, py = float(p["x"]), float(p["y"])
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)

        # ROLE-BASED THRESHOLD: Forwards are active from further away
        role = get_player_role(info)
        threshold = pressure_threshold_m + (2.0 if role == 'FW' else 0)

        if dist <= threshold:
            pressing_players.append({
                "player_id": pid,
                "name": info.get("name", ""),
                "number": info.get("number", ""),
                "role": role,
                "x": px,
                "y": py,
                "distance": dist
            })

    return pressing_players

# --- AGGREGATION ENGINE (The "Memory") ---

class PressingAggregator:
    """
    Accumulates pressing data over many frames to build the 'Average Network'.
    """
    def __init__(self):
        # Stores sum of X, Y coordinates to calculate average later
        self.player_positions = {}  # {pid: {'x_sum': 0, 'y_sum': 0, 'count': 0}}
        # Stores how many times two players pressed together
        self.partnerships = {}      # {(pid_A, pid_B): count}
        self.total_frames = 0

        # Outcome tracking (for Part 2: Pressing Outcome Analysis)
        self.player_outcomes = {}  # {pid: {'forced': 0, 'disrupted': 0, 'clean': 0}}
        self.partnership_outcomes = {}  # {(pid1, pid2): {'forced': 0, 'disrupted': 0}}

    def add_frame(self, pressing_players: List[Dict[str, Any]]):
        """Ingests a single frame of pressing data."""
        self.total_frames += 1
        
        # 1. Update Player Average Positions
        active_pids = []
        for p in pressing_players:
            pid = p['player_id']
            active_pids.append(pid)
            
            if pid not in self.player_positions:
                self.player_positions[pid] = {'x_sum': 0, 'y_sum': 0, 'count': 0}
            
            self.player_positions[pid]['x_sum'] += p['x']
            self.player_positions[pid]['y_sum'] += p['y']
            self.player_positions[pid]['count'] += 1

        # 2. Update Links (Who presses together?)
        active_pids.sort()
        for i in range(len(active_pids)):
            for j in range(i + 1, len(active_pids)):
                pair = (active_pids[i], active_pids[j])
                self.partnerships[pair] = self.partnerships.get(pair, 0) + 1

    def get_aggregated_data(self, min_participation_pct: float = 0.1):
        """
        Returns cleaned data for plotting.
        Filters out players/links that appear in < min_participation_pct of pressing actions.
        """
        if self.total_frames == 0:
            print("Warning: aggregated total_frames is 0")
            return {}, {}

        # Calculate Average Positions
        avg_positions = {}
        valid_pids = set()
        
        # Determine max participation for relative filtering
        # Or should we use total_frames? 
        # Usually, participation pct is relative to the total duration of the phase?
        # interpret min_participation_pct as % of total frames processed.
        min_count = self.total_frames * min_participation_pct
        
        # Correction: If identifying "average shape", we might want players who press FREQUENTLY.
        # if total_frames involves non-pressing moments, this threshold might be too high.
        # Assuming add_frame is only called when there IS pressing? 
        # Users code: "add_frame(links)". If links empty, frame is added but counts don't go up.
        # So total_frames tracks total observed time.
        
        # use max player count as reference instead of total_frames 
        # to ensure the "core" unit is kept even if pressing is rare overall?
        # The prompt says: "ignores random, one-off instances".
        
        counts = [d['count'] for d in self.player_positions.values()]
        max_player_count = max(counts) if counts else 0
        threshold = max_player_count * 0.1 # Dynamic threshold (10% of most active player)
        
        for pid, data in self.player_positions.items():
            if data['count'] > threshold: 
                avg = (data['x_sum'] / data['count'], data['y_sum'] / data['count'])
                avg_positions[pid] = avg
                valid_pids.add(pid)

        # Filter Links
        filtered_links = {}
        max_link_strength = max(self.partnerships.values()) if self.partnerships else 1
        
        for pair, count in self.partnerships.items():
            # Both players must be valid 
            if pair[0] in valid_pids and pair[1] in valid_pids:
                # Use simple relative strength for visualizing thickness
                filtered_links[pair] = count

        return avg_positions, filtered_links

    def get_aggregated_data_with_affinity(
        self,
        min_participation_pct: float = 0.1,
        affinity_method: str = "jaccard"
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Returns cleaned data with affinity scores for plotting.

        Args:
            min_participation_pct: Minimum participation threshold (0.0-1.0)
            affinity_method: "jaccard" or "cosine" similarity metric

        Returns:
            Tuple of (avg_positions, raw_counts, affinity_scores):
            - avg_positions: {pid: (x, y)} average pressing positions
            - raw_counts: {(pid1, pid2): count} raw co-press counts (for backward compatibility)
            - affinity_scores: {(pid1, pid2): score} normalized affinity scores (0.0-1.0)

        Example:
            >>> agg = PressingAggregator()
            >>> # ... add frames ...
            >>> positions, raw, affinity = agg.get_aggregated_data_with_affinity(method="jaccard")
        """
        from src.models.pressing_affinity import PressingAffinityCalculator

        # Get base aggregated data
        avg_positions, raw_counts = self.get_aggregated_data(min_participation_pct)

        # Calculate affinity scores using individual player press counts
        node_sizes = {pid: data['count'] for pid, data in self.player_positions.items()}
        affinity_scores = PressingAffinityCalculator.calculate_affinity_matrix(
            raw_counts, node_sizes, method=affinity_method
        )

        return avg_positions, raw_counts, affinity_scores

    def add_outcome(
        self,
        presser_ids: List[int],
        outcome: str  # 'forced_error', 'disrupted', or 'clean_escape'
    ):
        """
        Records an outcome for a pressing action.

        Args:
            presser_ids: List of player IDs involved in the pressing
            outcome: Outcome type ('forced_error', 'disrupted', 'clean_escape')
        """
        # Update player outcomes
        for pid in presser_ids:
            if pid not in self.player_outcomes:
                self.player_outcomes[pid] = {'forced': 0, 'disrupted': 0, 'clean': 0}

            if outcome == 'forced_error':
                self.player_outcomes[pid]['forced'] += 1
            elif outcome == 'disrupted':
                self.player_outcomes[pid]['disrupted'] += 1
            elif outcome == 'clean_escape':
                self.player_outcomes[pid]['clean'] += 1

        # Update partnership outcomes
        presser_ids_sorted = sorted(presser_ids)
        for i in range(len(presser_ids_sorted)):
            for j in range(i + 1, len(presser_ids_sorted)):
                pair = (presser_ids_sorted[i], presser_ids_sorted[j])

                if pair not in self.partnership_outcomes:
                    self.partnership_outcomes[pair] = {'forced': 0, 'disrupted': 0}

                if outcome == 'forced_error':
                    self.partnership_outcomes[pair]['forced'] += 1
                elif outcome == 'disrupted':
                    self.partnership_outcomes[pair]['disrupted'] += 1

    def get_outcome_statistics(self) -> Dict[str, Any]:
        """
        Returns aggregated outcome metrics.

        Returns:
            Dictionary with:
            - 'players': {pid: {'total': count, 'success_rate': float}}
            - 'partnerships': {(pid1, pid2): {'total': count, 'success_rate': float}}
            - 'overall_success_rate': float

        Example:
            >>> agg = PressingAggregator()
            >>> # ... add frames and outcomes ...
            >>> stats = agg.get_outcome_statistics()
            >>> print(f"Overall success: {stats['overall_success_rate']:.2%}")
        """
        player_stats = {}
        for pid, outcomes in self.player_outcomes.items():
            total = outcomes['forced'] + outcomes['disrupted'] + outcomes['clean']
            success = outcomes['forced'] + outcomes['disrupted']
            success_rate = success / total if total > 0 else 0.0

            player_stats[pid] = {
                'total': total,
                'forced_errors': outcomes['forced'],
                'disruptions': outcomes['disrupted'],
                'clean_escapes': outcomes['clean'],
                'success_rate': success_rate
            }

        partnership_stats = {}
        for pair, outcomes in self.partnership_outcomes.items():
            total = outcomes['forced'] + outcomes['disrupted']
            # Note: we don't track clean escapes for partnerships
            # Total should come from partnerships dict
            actual_total = self.partnerships.get(pair, 0)
            success_rate = total / actual_total if actual_total > 0 else 0.0

            partnership_stats[pair] = {
                'total': actual_total,
                'forced_errors': outcomes['forced'],
                'disruptions': outcomes['disrupted'],
                'success_rate': success_rate
            }

        # Calculate overall success rate
        all_totals = [s['total'] for s in player_stats.values()]
        all_successes = [s['forced_errors'] + s['disruptions'] for s in player_stats.values()]
        overall_success = sum(all_successes) / sum(all_totals) if sum(all_totals) > 0 else 0.0

        return {
            'players': player_stats,
            'partnerships': partnership_stats,
            'overall_success_rate': overall_success
        }


class TemporalPressingAggregator:
    """
    Aggregator that separates pressing by time phases.

    Divides each build-up into temporal phases (early/mid/late) to analyze
    how pressing behavior evolves during the build-up sequence.
    """

    def __init__(self):
        """Initialize separate aggregators for each phase."""
        self.early_agg = PressingAggregator()
        self.mid_agg = PressingAggregator()
        self.late_agg = PressingAggregator()
        self.build_up_durations = []  # Track for statistics

    def add_build_up(
        self,
        pressing_frames: List[Tuple[float, List[Dict[str, Any]]]],
        build_up_duration: float
    ):
        """
        Adds entire build-up, splitting into temporal phases.

        Args:
            pressing_frames: List of (timestamp_seconds, pressing_players) tuples
            build_up_duration: Total duration in seconds

        Example:
            >>> agg = TemporalPressingAggregator()
            >>> frames = [(0.5, players1), (1.2, players2), (2.8, players3)]
            >>> agg.add_build_up(frames, build_up_duration=3.0)
        """
        self.build_up_durations.append(build_up_duration)

        # Define phase boundaries (early: 0-33%, mid: 33-67%, late: 67-100%)
        early_cutoff = build_up_duration * 0.33
        late_start = build_up_duration * 0.67

        for timestamp, pressing_players in pressing_frames:
            if timestamp <= early_cutoff:
                self.early_agg.add_frame(pressing_players)
            elif timestamp >= late_start:
                self.late_agg.add_frame(pressing_players)
            else:
                self.mid_agg.add_frame(pressing_players)

    def get_phase_data(
        self,
        phase: str,
        min_participation_pct: float = 0.1
    ) -> Tuple[Dict, Dict]:
        """
        Returns data for specific phase.

        Args:
            phase: "early", "mid", or "late"
            min_participation_pct: Minimum participation threshold

        Returns:
            Tuple of (avg_positions, co_press_counts) for the specified phase

        Example:
            >>> early_pos, early_edges = agg.get_phase_data("early")
            >>> late_pos, late_edges = agg.get_phase_data("late")
        """
        if phase == "early":
            return self.early_agg.get_aggregated_data(min_participation_pct)
        elif phase == "late":
            return self.late_agg.get_aggregated_data(min_participation_pct)
        elif phase == "mid":
            return self.mid_agg.get_aggregated_data(min_participation_pct)
        else:
            raise ValueError(f"Unknown phase: {phase}. Must be 'early', 'mid', or 'late'")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the temporal aggregation.

        Returns:
            Dictionary with statistics like average duration, phase frame counts
        """
        return {
            'avg_duration': np.mean(self.build_up_durations) if self.build_up_durations else 0.0,
            'total_build_ups': len(self.build_up_durations),
            'early_frames': self.early_agg.total_frames,
            'mid_frames': self.mid_agg.total_frames,
            'late_frames': self.late_agg.total_frames
        }


def calculate_time_weighted_edges(
    pressing_events: List[Tuple[float, List[int]]],
    build_up_duration: float,
    weighting_mode: str = "linear_decay"
) -> Dict[Tuple[int, int], float]:
    """
    Calculates edge weights with temporal weighting.

    Earlier pressing is given higher weight to emphasize proactive pressing behavior.

    Args:
        pressing_events: List of (timestamp, [presser_ids]) tuples
        build_up_duration: Total duration in seconds
        weighting_mode: Weighting strategy
            - "linear_decay": w(t) = 1 - (t / duration)
            - "exponential_decay": w(t) = exp(-2*t/duration)
            - "early_boost": w(t) = 2.0 if t < 33% else 0.5

    Returns:
        Dictionary of {(pid1, pid2): weighted_count}

    Example:
        >>> events = [(0.5, [1,2,3]), (2.5, [1,2])]
        >>> weights = calculate_time_weighted_edges(events, 3.0, "linear_decay")
        >>> # Frame at t=0.5 gets weight ~0.83, frame at t=2.5 gets weight ~0.17
    """
    weighted_edges = {}

    for timestamp, presser_ids in pressing_events:
        # Calculate weight based on mode
        if weighting_mode == "linear_decay":
            # Linearly decrease weight over time (early = high, late = low)
            weight = max(1.0 - (timestamp / build_up_duration), 0.1)
        elif weighting_mode == "exponential_decay":
            # Exponentially decrease weight
            lambda_param = 2.0 / build_up_duration
            weight = np.exp(-lambda_param * timestamp)
        elif weighting_mode == "early_boost":
            # Binary: high weight for early, low for late
            weight = 2.0 if timestamp < (0.33 * build_up_duration) else 0.5
        else:
            # Default: uniform weighting
            weight = 1.0

        # Add weighted co-press counts for all pairs
        presser_ids = sorted(presser_ids)
        for i in range(len(presser_ids)):
            for j in range(i + 1, len(presser_ids)):
                pair = (presser_ids[i], presser_ids[j])
                weighted_edges[pair] = weighted_edges.get(pair, 0.0) + weight

    return weighted_edges


def identify_trigger_player(
    frames: List[Dict[str, Any]],
    player_map: Dict[int, Dict[str, Any]],
    rm_team_id: int,
    pressure_threshold_m: float = 5.0,
    kick_time: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Identify the first Real Madrid player to apply pressure on the ball carrier.

    The trigger player is the RM player who first gets within pressure_threshold_m
    of the ball carrier, initiating the pressing action.

    Args:
        frames: List of frame dictionaries with player and ball data
        player_map: Mapping of player_id to player metadata (name, number, team_id, etc.)
        rm_team_id: Team ID for Real Madrid
        pressure_threshold_m: Distance threshold for pressure (default 5.0m)
        kick_time: Optional kick timestamp to calculate trigger timing

    Returns:
        Dictionary with trigger player info (player_id, name, number, timestamp, distance)
        or None if no trigger detected
    """
    if not frames:
        return None

    # Find first frame with pressure
    for frame in frames:
        # Get ball carrier
        carrier_id = None
        carrier_pos = None
        ball_data = frame.get("ball_data", {})

        # Try to find carrier by proximity to ball
        if ball_data.get("x") is not None and ball_data.get("y") is not None:
            bx, by = float(ball_data["x"]), float(ball_data["y"])

            min_dist = float('inf')
            for p in frame.get("player_data", []):
                pid = int(p["player_id"])
                info = player_map.get(pid)
                if not info or info.get("team_id") == rm_team_id:
                    continue  # Skip RM players

                px, py = p.get("x"), p.get("y")
                if px is None or py is None:
                    continue

                dist = np.sqrt((float(px) - bx)**2 + (float(py) - by)**2)
                if dist < min_dist and dist <= 8.0:  # Within 8.0m = has possession (goal-kick scenario)
                    min_dist = dist
                    carrier_id = pid
                    carrier_pos = (float(px), float(py))

        if not carrier_pos:
            continue

        # Check RM players for pressure
        cx, cy = carrier_pos
        closest_rm_player = None
        closest_rm_dist = float('inf')

        for p in frame.get("player_data", []):
            pid = int(p["player_id"])
            info = player_map.get(pid)
            if not info or info.get("team_id") != rm_team_id:
                continue  # Only RM players

            px, py = p.get("x"), p.get("y")
            if px is None or py is None:
                continue

            dist = np.sqrt((float(px) - cx)**2 + (float(py) - cy)**2)

            # Role-based threshold: Forwards get +2m range
            role = get_player_role(info)
            threshold = pressure_threshold_m + (2.0 if role == 'FW' else 0)

            if dist <= threshold and dist < closest_rm_dist:
                closest_rm_dist = dist
                closest_rm_player = {
                    "player_id": pid,
                    "name": info.get("name", "Unknown"),
                    "number": info.get("number", "?"),
                    "role": role,
                    "timestamp": str(frame.get("timestamp", "")),
                    "distance": dist
                }

        if closest_rm_player:
            return closest_rm_player

    return None

# To maintain compatibility with app.py which imports:
# calculate_pressing_links, create_network_traces, create_pressure_zone_shapes, get_ball_carrier_for_frame, identify_trigger_player

# I must add back the functions required by app.py that were NOT in the user snippet, 
# or else app.py will crash.
# The user said "Replace your current file with this", but user might not realize app.py dependency.
# I will add the missing compatibility functions at the bottom.

def get_ball_carrier_for_frame(
    frame_data: Dict[str, Any],
    player_map: Dict[int, Dict[str, Any]],
    rm_team_id: int,
    opponent_team_id: int,
    possession_radius: float = 8.0,  # Increased from 1.5m to 8.0m for goal-kick scenarios
) -> Optional[int]:
    """Identify ball carrier by finding closest opponent player to ball.

    Args:
        frame_data: Frame dictionary with player and ball data
        player_map: Mapping of player_id to metadata
        rm_team_id: Real Madrid team ID (to exclude from carrier search)
        opponent_team_id: Opponent team ID (potential carriers)
        possession_radius: Maximum distance for possession (default 8.0m for goal kicks)

    Returns:
        Player ID of ball carrier, or None if no opponent within radius
    """
    ball = frame_data.get("ball_data", {})
    bx = ball.get("x")
    by = ball.get("y")
    if bx is None or by is None:
        return None

    min_dist = float('inf')
    carrier_id = None

    for p in frame_data.get("player_data", []):
        pid = int(p["player_id"])
        info = player_map.get(pid, {})
        if info.get("team_id") != opponent_team_id:
            continue

        px, py = p.get("x"), p.get("y")
        if px is None or py is None:
            continue

        dist = np.sqrt((float(px) - float(bx))**2 + (float(py) - float(by))**2)
        if dist < min_dist:
            min_dist = dist
            if dist <= possession_radius:
                carrier_id = pid

    return carrier_id

def create_network_traces(
    pressing_players: List[Dict[str, Any]],
    carrier_pos: Tuple[float, float],
    show_network: bool = True,
) -> List[Any]: # Returns Plotly traces
    """Compatibility: Create Plotly traces for Streamlit app."""
    # This was used in app.py. The new file uses Matplotlib.
    # I need to reimplement a basic version for app.py to work.
    import plotly.graph_objects as go
    if not show_network or not pressing_players: return []
    traces = []
    cx, cy = carrier_pos
    for presser in pressing_players:
        role = presser.get("role", "Unknown")
        color = "rgba(255, 50, 50, 0.6)" # Default Red
        if role == "MF": color = "rgba(255, 165, 0, 0.6)"
        elif role == "DF": color = "rgba(50, 50, 255, 0.6)"
        
        traces.append(go.Scatter(
            x=[presser["x"], cx], y=[presser["y"], cy],
            mode="lines", line=dict(color=color, width=2),
            hoverinfo="skip", showlegend=False
        ))
    return traces

def create_pressure_zone_shapes(
    carrier_pos: Tuple[float, float],
    pressure_threshold_m: float = 5.0,
    show_zones: bool = True,
) -> List[Dict[str, Any]]:
    """Compatibility: Pressure zones for app.py (deprecated - use create_pressure_zone_traces)."""
    if not show_zones or not carrier_pos: return []
    cx, cy = carrier_pos
    return [{
        "type": "circle",
        "x0": cx - pressure_threshold_m, "y0": cy - pressure_threshold_m,
        "x1": cx + pressure_threshold_m, "y1": cy + pressure_threshold_m,
        "line": {"color": "rgba(255, 200, 0, 0.3)", "width": 2, "dash": "dot"},
        "fillcolor": "rgba(255, 200, 0, 0.1)",
    }]

def create_pressure_zone_traces(
    carrier_pos: Optional[Tuple[float, float]],
    pressure_threshold_m: float = 5.0,
    show_zones: bool = True,
) -> List[Any]:
    """Create animated pressure zone traces (circle around ball carrier).

    Returns scatter traces that can be updated in animation frames, unlike shapes
    which are part of the static layout.

    Args:
        carrier_pos: (x, y) position of ball carrier, or None if no carrier
        pressure_threshold_m: Radius of pressure zone in meters
        show_zones: Whether to show zones

    Returns:
        List of Plotly scatter traces (empty if no carrier or zones disabled)
    """
    import plotly.graph_objects as go

    if not show_zones or not carrier_pos:
        return []

    cx, cy = carrier_pos

    # Create circle points
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = cx + pressure_threshold_m * np.cos(theta)
    y_circle = cy + pressure_threshold_m * np.sin(theta)

    # Create filled circle trace
    circle_trace = go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        fill='toself',
        fillcolor='rgba(255, 200, 0, 0.15)',
        line=dict(color='rgba(255, 200, 0, 0.4)', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False,
    )

    return [circle_trace]
