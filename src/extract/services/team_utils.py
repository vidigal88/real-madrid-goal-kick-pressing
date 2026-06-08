"""Team and player identification utilities.

This module provides functions for identifying teams (specifically Real Madrid)
and finding specific players like goalkeepers from match metadata.
"""

from __future__ import annotations

from typing import Any


def find_real_madrid_and_opponent(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identify Real Madrid team and their opponent from match metadata.

    Searches for Real Madrid by:
    1. Team name containing "real madrid" (case-insensitive)
    2. Team acronym being "REA" or "RMA"

    Args:
        meta: Match metadata dictionary containing home_team and away_team

    Returns:
        Tuple of (real_madrid_team_dict, opponent_team_dict)

    Raises:
        ValueError: If Real Madrid cannot be identified in the metadata

    Examples:
        >>> meta = {
        ...     "home_team": {"id": 1, "name": "Real Madrid", "acronym": "RMA"},
        ...     "away_team": {"id": 2, "name": "Barcelona", "acronym": "BAR"}
        ... }
        >>> rm, opp = find_real_madrid_and_opponent(meta)
        >>> rm["name"]
        'Real Madrid'
        >>> opp["name"]
        'Barcelona'
    """
    home = meta["home_team"]
    away = meta["away_team"]

    # Check by name
    if "real madrid" in (home.get("name") or "").lower():
        return home, away
    if "real madrid" in (away.get("name") or "").lower():
        return away, home

    # Check by acronym
    if (home.get("acronym") or "").upper() in {"REA", "RMA"}:
        return home, away
    if (away.get("acronym") or "").upper() in {"REA", "RMA"}:
        return away, home

    raise ValueError("Could not identify Real Madrid team in meta data")


def find_starting_goalkeeper(meta: dict[str, Any], team_id: int) -> dict[str, Any] | None:
    """Find the starting goalkeeper for a given team.

    Args:
        meta: Match metadata dictionary containing players list
        team_id: Team identifier to search for

    Returns:
        Player dictionary if found, None otherwise. A starting goalkeeper is defined as:
        - team_id matches
        - player_role.acronym is "GK"
        - start_time is "00:00:00"

    Examples:
        >>> meta = {
        ...     "players": [
        ...         {
        ...             "id": 1,
        ...             "team_id": 10,
        ...             "player_role": {"acronym": "GK"},
        ...             "start_time": "00:00:00",
        ...             "short_name": "Courtois"
        ...         },
        ...         {
        ...             "id": 2,
        ...             "team_id": 10,
        ...             "player_role": {"acronym": "DF"},
        ...             "start_time": "00:00:00",
        ...             "short_name": "Ramos"
        ...         }
        ...     ]
        ... }
        >>> gk = find_starting_goalkeeper(meta, 10)
        >>> gk["short_name"]
        'Courtois'
    """
    for p in meta.get("players", []):
        if (
            p.get("team_id") == team_id
            and (p.get("player_role") or {}).get("acronym") == "GK"
            and p.get("start_time") == "00:00:00"
        ):
            return p
    return None
