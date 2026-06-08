"""
Pressing Outcome Detection

Analyzes the effectiveness of pressing actions by tracking what happens after
pressing moments: forced errors (turnovers), disruptions (backward/sideways passes),
or clean escapes (successful progression).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np


class PressingOutcome(Enum):
    """Possible outcomes of a pressing action."""
    FORCED_ERROR = "forced_error"      # Ball turnover within 2s
    DISRUPTED = "disrupted"            # Backward/sideways pass within 3s
    CLEAN_ESCAPE = "clean_escape"      # Forward progression
    UNKNOWN = "unknown"                # Insufficient data


@dataclass
class PressingEvent:
    """Represents a single pressing event with its outcome."""
    frame: int
    timestamp: str
    presser_ids: List[int]             # All players involved in pressing
    ball_carrier_id: int
    carrier_position: Tuple[float, float]
    outcome: PressingOutcome
    outcome_time: Optional[str] = None # When outcome was determined
    ball_progression_m: float = 0.0    # Forward distance traveled


@dataclass
class PlayerPressingStats:
    """Aggregated pressing statistics for a single player."""
    player_id: int
    total_presses: int = 0
    forced_errors: int = 0
    disruptions: int = 0
    clean_escapes: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate: (forced_errors + disruptions) / total"""
        if self.total_presses == 0:
            return 0.0
        return (self.forced_errors + self.disruptions) / self.total_presses


@dataclass
class EdgePressingStats:
    """Pressing statistics for a player partnership."""
    player_pair: Tuple[int, int]
    co_press_count: int = 0
    forced_errors: int = 0
    disruptions: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate for this partnership."""
        if self.co_press_count == 0:
            return 0.0
        return (self.forced_errors + self.disruptions) / self.co_press_count


def detect_pressing_outcome(
    df_norm: pd.DataFrame,
    pressing_frame_idx: int,
    ball_carrier_id: int,
    defending_team_id: int,
    lookforward_frames: int = 20  # 2 seconds at 10Hz
) -> PressingOutcome:
    """
    Analyzes frames AFTER a pressing action to determine outcome.

    Args:
        df_norm: Normalized dataframe with ball carrier info
        pressing_frame_idx: Index of the pressing frame
        ball_carrier_id: ID of the player being pressed
        defending_team_id: ID of the defending (pressing) team
        lookforward_frames: How many frames ahead to check (default 20 = 2s at 10Hz)

    Returns:
        PressingOutcome indicating success/failure

    Logic:
        1. FORCED_ERROR if:
           - Ball carrier changes to defending team player
           - Ball goes significantly backward (regression >5m)
        2. DISRUPTED if:
           - Ball moves backward (Y decreases >3m)
           - Ball moves sideways (lateral movement > forward movement)
        3. CLEAN_ESCAPE if:
           - Ball progresses forward >10m
           - Ball carrier maintains possession and advances
    """
    # Get unique frame identifiers in the dataframe
    frames = df_norm['frame'].unique()
    current_frame = frames[pressing_frame_idx] if pressing_frame_idx < len(frames) else frames[-1]

    # Find current ball carrier position
    carrier_rows = df_norm[(df_norm['frame'] == current_frame) & (df_norm['player_id'] == ball_carrier_id)]
    if carrier_rows.empty:
        return PressingOutcome.UNKNOWN

    initial_y = carrier_rows['y_norm'].iloc[0]
    initial_x = carrier_rows['x_norm'].iloc[0]

    # Look at next N frames
    future_frames = frames[pressing_frame_idx + 1:min(pressing_frame_idx + 1 + lookforward_frames, len(frames))]

    if len(future_frames) == 0:
        return PressingOutcome.UNKNOWN

    for future_frame in future_frames:
        future_df = df_norm[df_norm['frame'] == future_frame]

        # Check for possession change (forced error)
        if 'ball_carrier_id' in future_df.columns:
            future_carriers = future_df[future_df['ball_carrier_id'] == 1.0]
            if not future_carriers.empty:
                new_carrier_id = future_carriers['player_id'].iloc[0]
                new_carrier_team = future_carriers['team_id'].iloc[0] if 'team_id' in future_carriers.columns else None

                # If ball carrier changed to defending team = forced error
                if new_carrier_team == defending_team_id:
                    return PressingOutcome.FORCED_ERROR

                # Track ball position for the new carrier
                if new_carrier_id != ball_carrier_id:
                    new_carrier_y = future_carriers['y_norm'].iloc[0]
                    new_carrier_x = future_carriers['x_norm'].iloc[0]

                    y_delta = new_carrier_y - initial_y
                    x_delta = abs(new_carrier_x - initial_x)

                    # Significant backward movement = disrupted
                    if y_delta < -3.0:
                        return PressingOutcome.DISRUPTED

                    # More lateral than forward = disrupted
                    if x_delta > abs(y_delta) and y_delta < 5.0:
                        return PressingOutcome.DISRUPTED

                    # Strong forward progression = clean escape
                    if y_delta > 10.0:
                        return PressingOutcome.CLEAN_ESCAPE

    # Default: check final position
    final_frame = future_frames[-1]
    final_df = df_norm[df_norm['frame'] == final_frame]

    if 'ball_carrier_id' in final_df.columns:
        final_carriers = final_df[final_df['ball_carrier_id'] == 1.0]
        if not final_carriers.empty:
            final_y = final_carriers['y_norm'].iloc[0]
            y_progression = final_y - initial_y

            if y_progression < -3.0:
                return PressingOutcome.DISRUPTED
            elif y_progression > 10.0:
                return PressingOutcome.CLEAN_ESCAPE

    # Inconclusive
    return PressingOutcome.UNKNOWN


def aggregate_outcome_statistics(
    events: List[PressingEvent]
) -> Tuple[Dict[int, PlayerPressingStats], Dict[Tuple[int, int], EdgePressingStats]]:
    """
    Aggregates pressing events into player and partnership statistics.

    Args:
        events: List of PressingEvent objects

    Returns:
        Tuple of (player_stats, partnership_stats):
        - player_stats: {player_id: PlayerPressingStats}
        - partnership_stats: {(pid1, pid2): EdgePressingStats}

    Example:
        >>> events = [...]  # List of pressing events
        >>> player_stats, edge_stats = aggregate_outcome_statistics(events)
        >>> tchouameni_stats = player_stats[123]
        >>> print(f"Success rate: {tchouameni_stats.success_rate:.2%}")
    """
    player_stats: Dict[int, PlayerPressingStats] = {}
    partnership_stats: Dict[Tuple[int, int], EdgePressingStats] = {}

    for event in events:
        # Update player statistics
        for pid in event.presser_ids:
            if pid not in player_stats:
                player_stats[pid] = PlayerPressingStats(player_id=pid)

            player_stats[pid].total_presses += 1

            if event.outcome == PressingOutcome.FORCED_ERROR:
                player_stats[pid].forced_errors += 1
            elif event.outcome == PressingOutcome.DISRUPTED:
                player_stats[pid].disruptions += 1
            elif event.outcome == PressingOutcome.CLEAN_ESCAPE:
                player_stats[pid].clean_escapes += 1

        # Update partnership statistics
        presser_ids = sorted(event.presser_ids)
        for i in range(len(presser_ids)):
            for j in range(i + 1, len(presser_ids)):
                pair = (presser_ids[i], presser_ids[j])

                if pair not in partnership_stats:
                    partnership_stats[pair] = EdgePressingStats(player_pair=pair)

                partnership_stats[pair].co_press_count += 1

                if event.outcome == PressingOutcome.FORCED_ERROR:
                    partnership_stats[pair].forced_errors += 1
                elif event.outcome == PressingOutcome.DISRUPTED:
                    partnership_stats[pair].disruptions += 1

    return player_stats, partnership_stats


def export_outcome_statistics(
    player_stats: Dict[int, PlayerPressingStats],
    player_names: Dict[int, str],
    output_path: str
) -> None:
    """
    Exports outcome statistics to JSON file.

    Args:
        player_stats: Player pressing statistics
        player_names: Mapping of player IDs to names
        output_path: Path to save JSON file
    """
    import json

    output = {
        'players': [],
        'summary': {
            'total_players': len(player_stats),
            'avg_success_rate': np.mean([s.success_rate for s in player_stats.values()]) if player_stats else 0.0
        }
    }

    for pid, stats in sorted(player_stats.items(), key=lambda x: x[1].success_rate, reverse=True):
        output['players'].append({
            'player_id': pid,
            'player_name': player_names.get(pid, str(pid)),
            'total_presses': stats.total_presses,
            'forced_errors': stats.forced_errors,
            'disruptions': stats.disruptions,
            'clean_escapes': stats.clean_escapes,
            'success_rate': round(stats.success_rate, 3)
        })

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
