"""Goal kick detection and analysis utilities.

This module handles detection of goal kick events from dynamic event data
and analysis of goalkeeper positioning for goal kicks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GoalKickRef:
    """Reference to a goal kick event from dynamic data.

    Attributes:
        game_id: Match identifier
        period: Match period (1, 2, etc.)
        event_id: Unique event identifier
        time_start: Event start time in dynamic format (MM:SS.S)
        team_id: Team taking the goal kick
    """
    game_id: str
    period: int
    event_id: str
    time_start: str  # dynamic time "MM:SS.S"
    team_id: int


def filter_goal_kick_refs(
    dynamic_df: pd.DataFrame,
    opponent_team_id: int,
    game_id: str
) -> list[GoalKickRef]:
    """Extract goal kick references for a specific team from dynamic event data.

    Filters for events where:
    - game_interruption_before is "goal_kick_for"
    - team_id matches the specified opponent

    Args:
        dynamic_df: DataFrame containing dynamic event data
        opponent_team_id: ID of the team taking goal kicks
        game_id: Match identifier to include in references

    Returns:
        List of GoalKickRef objects for all matching goal kicks

    Examples:
        >>> dynamic_df = pd.DataFrame({
        ...     "game_interruption_before": ["goal_kick_for", "throw_in", "goal_kick_for"],
        ...     "team_id": [10, 10, 20],
        ...     "event_id": ["e1", "e2", "e3"],
        ...     "time_start": ["5:30.2", "6:15.5", "8:45.1"],
        ...     "period": [1, 1, 1]
        ... })
        >>> refs = filter_goal_kick_refs(dynamic_df, 10, "2014987")
        >>> len(refs)
        1
        >>> refs[0].time_start
        '5:30.2'
    """
    df = dynamic_df[
        (dynamic_df["game_interruption_before"] == "goal_kick_for") & (dynamic_df["team_id"] == opponent_team_id)
    ][["event_id", "time_start", "period", "team_id"]].copy()

    refs: list[GoalKickRef] = []
    for _, r in df.iterrows():
        refs.append(
            GoalKickRef(
                game_id=game_id,
                period=int(r["period"]),
                event_id=str(r["event_id"]),
                time_start=str(r["time_start"]),
                team_id=int(r["team_id"]),
            )
        )
    return refs


def detect_goalkeeper_side_from_parquet(
    tracking_df: pd.DataFrame,
    gk_id: int,
    period: int
) -> str | None:
    """Determine which side of the pitch the goalkeeper is defending.

    Analyzes goalkeeper X-coordinate positions throughout a period to determine
    if they're defending the left (negative X) or right (positive X) goal.

    Args:
        tracking_df: DataFrame with tracking data
        gk_id: Player ID of the goalkeeper
        period: Match period to analyze

    Returns:
        "left" if goalkeeper defends left side (negative X),
        "right" if goalkeeper defends right side (positive X),
        None if no valid tracking data found

    Examples:
        >>> tracking_df = pd.DataFrame({
        ...     "period": [1, 1, 1, 1],
        ...     "player_id": [1, 1, 1, 2],
        ...     "is_detected": [True, True, True, True],
        ...     "x": [-48.0, -49.5, -47.2, 10.0]
        ... })
        >>> detect_goalkeeper_side_from_parquet(tracking_df, 1, 1)
        'left'
    """
    df = tracking_df[
        (tracking_df["period"] == period) & (tracking_df["player_id"] == gk_id) & (tracking_df["is_detected"] == True)
    ][["x"]]
    if df.empty:
        return None
    avg_x = float(df["x"].mean())
    return "left" if avg_x < 0 else "right"
