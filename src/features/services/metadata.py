"""Game metadata loading and enrichment utilities.

This module provides functions to load SkillCorner game metadata from JSON
files and enrich tracking DataFrames with team_id mappings based on player
assignments.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict


def load_game_meta(game_id: int, raw_root: Path) -> Dict[str, Any]:
    """Load game metadata including player and team information from JSON.

    Args:
        game_id: Unique identifier for the game/match.
        raw_root: Root directory path containing 'meta/' subdirectory with
            JSON metadata files.

    Returns:
        Dictionary containing game metadata with keys:
            - 'players': List of player dictionaries with 'id', 'team_id', 'name'.
            - 'teams': List of team dictionaries.
            - Additional match-level metadata fields.

    Raises:
        FileNotFoundError: If metadata JSON file doesn't exist for the game_id.

    Example:
        >>> meta = load_game_meta(4039, Path("data/raw/RealMadrid"))
        >>> print(len(meta['players']))
        26
    """
    path = raw_root / "meta" / f"{game_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found for game {game_id}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_player_team_map(meta: Dict[str, Any]) -> Dict[int, int]:
    """Build player-to-team mapping from game metadata.

    Args:
        meta: Game metadata dictionary from load_game_meta(), containing
            'players' list with 'id' and 'team_id' fields.

    Returns:
        Dictionary mapping player_id (int) to team_id (int).

    Example:
        >>> meta = load_game_meta(4039, Path("data/raw/RealMadrid"))
        >>> player_map = build_player_team_map(meta)
        >>> print(player_map[123456])  # player_id -> team_id
        78
    """
    mapping = {}
    for p in meta.get("players", []):
        mapping[int(p["id"])] = int(p["team_id"])
    return mapping


def enrich_with_team_id(df: pd.DataFrame, game_id: int,
                        raw_root: str = "data/raw/RealMadrid") -> pd.DataFrame:
    """Add team_id column to tracking DataFrame based on player assignments.

    Args:
        df: Tracking DataFrame with 'player_id' column.
        game_id: Unique identifier for the game/match.
        raw_root: Root directory path containing game metadata files.

    Returns:
        Copy of input DataFrame with added 'team_id' column. Ball rows
        (player_id=NaN) will have team_id=NaN.

    Raises:
        FileNotFoundError: If metadata file doesn't exist for the game_id.
        KeyError: If 'player_id' column is missing from df.

    Example:
        >>> df_enriched = enrich_with_team_id(df, game_id=4039)
        >>> print(df_enriched[['player_id', 'team_id']].head())
           player_id  team_id
        0     123456       78
        1     123457       78
        2        NaN      NaN  # ball row
    """
    raw_path = Path(raw_root)
    meta = load_game_meta(game_id, raw_path)
    player_map = build_player_team_map(meta)
    
    # Map
    # Ensure player_id is int
    if "player_id" in df.columns:
        df = df.copy()
        # Handle NaN player_ids (ball)
        # We only map valid players.
        # Create a series to map
        
        # Performance: map is fast
        df["team_id"] = df["player_id"].map(player_map)
        # Ball remains NaN team_id
        
    return df
