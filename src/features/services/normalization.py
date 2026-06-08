"""Coordinate normalization for consistent pitch orientation.

This module provides coordinate system normalization to ensure that the opponent
(team with the ball during goal kicks) always attacks from left to right in the
normalized coordinate system. This standardization is critical for comparing
pressing patterns across different periods and matches.
"""

from __future__ import annotations

import pandas as pd
from src.features.config import NormalizationConfig


def normalize_coordinates(
    df: pd.DataFrame,
    gk_side: str,
    config: NormalizationConfig = NormalizationConfig()
) -> pd.DataFrame:
    """Normalize pitch coordinates to standard orientation (opponent attacks right).

    Transforms tracking coordinates so that the opponent goalkeeper is always on the
    left side of the pitch (x ≈ -52.5) and the opponent team attacks toward the right
    (positive x direction). This ensures consistent spatial analysis regardless of
    which half the goal kick occurred in.

    The transformation logic:
    - If gk_side == "left": Goalkeeper on left, team attacks right → No transformation
    - If gk_side == "right": Goalkeeper on right, team attacks left → Flip both x and y coordinates (multiply by -1)

    The function automatically detects and normalizes common coordinate column pairs:
    - ('x', 'y'): Standard player/ball coordinates
    - ('ball_x', 'ball_y'): Explicit ball coordinates

    Args:
        df: Tracking data DataFrame containing coordinate columns. Expected to be in
            long format with columns like 'x', 'y' for player/ball positions, or wide
            format with 'ball_x', 'ball_y' columns. The DataFrame is copied and not
            modified in place.
        gk_side: Side of the pitch where the goalkeeper is positioned. Must be either
            "left" or "right". This is typically extracted during build-up detection
            based on the goalkeeper's position at the ready moment.
        config: Normalization configuration containing pitch dimensions. Defaults to
            standard values (105m × 68m pitch).

    Returns:
        DataFrame with normalized coordinate columns added:
        - 'x_norm', 'y_norm': Normalized coordinates for standard 'x', 'y' columns
        - 'ball_x_norm', 'ball_y_norm': Normalized ball coordinates if 'ball_x', 'ball_y' exist

        Original coordinate columns are preserved unchanged.

    Raises:
        ValueError: If gk_side is not "left" or "right".

    Example:
        >>> import pandas as pd
        >>> from src.features.services.normalization import normalize_coordinates
        >>> from src.features.config import NormalizationConfig
        >>>
        >>> # Sample tracking data with goalkeeper on right side
        >>> df = pd.DataFrame({
        ...     'time': ['00:00:01', '00:00:02'],
        ...     'x': [45.0, 40.0],      # Opponent players near right side
        ...     'y': [5.0, -3.0],
        ...     'ball_x': [48.0, 46.0],
        ...     'ball_y': [1.0, 2.0]
        ... })
        >>>
        >>> # Normalize so opponent attacks right
        >>> df_norm = normalize_coordinates(df, gk_side="right")
        >>> print(df_norm[['x', 'x_norm', 'ball_x', 'ball_x_norm']])
             x  x_norm  ball_x  ball_x_norm
        0  45.0   -45.0    48.0        -48.0
        1  40.0   -40.0    46.0        -46.0
        >>>
        >>> # Now goalkeeper is on left (negative x), opponent attacks right

    Notes:
        - The standard SkillCorner coordinate system has pitch center at (0, 0),
          with pitch ends at (-52.5, 0) and (52.5, 0).
        - After normalization, the opponent goalkeeper is always at approximately
          x_norm ≈ -52.5, and the opponent attacks toward x_norm ≈ +52.5.
        - This normalization is essential for:
          * Computing meaningful spatial features (e.g., forward pass distance)
          * Aggregating pressing patterns across multiple build-ups
          * Visualizing consistent heatmaps and zone assignments
    """
    df = df.copy()
    
    # Check for x and y columns
    # try to normalize common coordinate columns
    cols_to_normalize = []
    if 'x' in df.columns and 'y' in df.columns:
        cols_to_normalize.append(('x', 'y'))
    if 'ball_x' in df.columns and 'ball_y' in df.columns:
        cols_to_normalize.append(('ball_x', 'ball_y'))
        
    # Also handle player specific columns if it's wide format (e.g. x_1, y_1)
    # assume standard 'x', 'y' for now or 'ball_x', 'ball_y'.
    
    if gk_side == "left":
        # Already attacking right (GK on left)
        for x_col, y_col in cols_to_normalize:
            df[f"{x_col}_norm"] = df[x_col]
            df[f"{y_col}_norm"] = df[y_col]
            
    elif gk_side == "right":
        # Attack left -> Flip to right
        for x_col, y_col in cols_to_normalize:
            df[f"{x_col}_norm"] = -df[x_col]
            df[f"{y_col}_norm"] = -df[y_col]
            
    else:
        raise ValueError(f"Invalid gk_side: {gk_side}. Must be 'left' or 'right'.")

    return df
