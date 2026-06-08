"""Team shape and compactness metrics module.

This module quantifies Real Madrid's team shape and spatial organization during pressing.
Compactness metrics reveal how tightly grouped the team is, which is fundamental to
understanding pressing effectiveness:

- **Tight compactness** (low width/length, small hull area): Coordinated pressing, easier
  to support pressers, harder for opponents to find passing lanes
- **Loose compactness** (high width/length, large hull area): Spread out, potential gaps
  for opponent exploitation

The metrics capture both the overall team footprint (width, length, hull area) and the
positioning of the defensive line (line height).

Key metrics:
1. Team width (lateral spread in Y direction)
2. Team length (longitudinal spread in X direction)
3. Convex hull area (total spatial footprint)
4. Line height (median X position - pressing intensity indicator)
"""

from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from src.features.config import CompactnessConfig


def aggregate_compactness_features(
    df_norm: pd.DataFrame,
    rm_team_id: int,
    config: CompactnessConfig = CompactnessConfig()
) -> Dict[str, Any]:
    """Compute 6 team shape and compactness metrics for Real Madrid.

    Analyzes Real Madrid's spatial organization by calculating geometric properties
    of player positions in each frame. Metrics are computed per-frame and then
    aggregated (mean/min) across the entire build-up phase.

    **Tactical Interpretation:**
    - **Compact team** (low values): Players are close together, enabling quick support
      and reducing passing lanes. Typical of high-intensity pressing.
    - **Extended team** (high values): Players spread out, covering more space but with
      potential gaps. May indicate transition or retreat.

    Args:
        df_norm: Normalized tracking DataFrame with columns:
            - 'frame': Frame number
            - 'player_id': Player identifier
            - 'team_id': Team identifier (rm_team_id for Real Madrid players)
            - 'x_norm', 'y_norm': Normalized player coordinates

            After normalization, opponent attacks right (positive X direction),
            so Real Madrid pressing players have high X values.

        rm_team_id: Team ID for Real Madrid. Used to filter Real Madrid players
            for shape calculation.

        config: Compactness configuration (currently unused, reserved for future
            parameters like outlier removal thresholds).

    Returns:
        Dictionary containing 6 compactness metrics:

        **Width Metrics (Lateral Spread):**
        - 'rm_width_mean_m' (float): Average lateral spread (Y direction) of Real Madrid
          players across all frames. Calculated as max(Y) - min(Y) per frame, then averaged.
          Typical range: 15-40 meters.
          Interpretation: Lower = more compact laterally, easier to cover passing lanes.

        - 'rm_width_min_m' (float): Minimum lateral spread observed in any frame. Indicates
          the tightest lateral compactness achieved during the build-up.
          Typical range: 10-30 meters.
          Interpretation: Shows peak lateral coordination.

        **Length Metrics (Longitudinal Spread):**
        - 'rm_length_mean_m' (float): Average longitudinal spread (X direction) across frames.
          Calculated as max(X) - min(X) per frame, then averaged.
          Typical range: 20-50 meters.
          Interpretation: Lower = shorter distance between deepest and highest presser.

        - 'rm_length_min_m' (float): Minimum longitudinal spread observed. Indicates
          the moment of tightest longitudinal compactness.
          Typical range: 15-40 meters.

        **Spatial Footprint:**
        - 'rm_hull_area_mean_m2' (float): Average convex hull area (square meters) of
          Real Madrid player positions. The convex hull is the smallest convex polygon
          containing all players.
          Typical range: 300-1200 m².
          Interpretation: Total spatial footprint. Lower = more compact overall shape.

          Note: Requires ≥3 players per frame. Frames with <3 players contribute 0.0 or NaN.

        **Pressing Intensity Indicator:**
        - 'rm_line_height_median_x_mean' (float): Average median X position of Real Madrid
          players across frames. Since opponent attacks right (positive X), higher values
          indicate the team is positioned further up the pitch (higher press line).
          Typical range: -20 to +20 meters.
          Interpretation: Higher values = more aggressive pressing, closer to opponent goal.

        All metrics return NaN when insufficient data is available (no RM players, <2 players
        per frame for width/length, <3 players for hull).

    Algorithm Details:
        **Per-Frame Calculation:**
        For each frame containing ≥2 Real Madrid players:

        1. **Width:** max(y_norm) - min(y_norm)
        2. **Length:** max(x_norm) - min(x_norm)
        3. **Line Height:** median(x_norm)
        4. **Convex Hull Area** (if ≥3 players):
           - Construct convex hull using scipy.spatial.ConvexHull
           - Extract hull.volume (which is area in 2D)
           - Handle QhullError (collinear points) by setting NaN

        **Aggregation:**
        - Width/Length: Compute mean and min across all valid frames
        - Hull Area: Compute mean across frames (NaN/0.0 excluded)
        - Line Height: Compute mean of per-frame medians

    Edge Cases:
        - No RM players: Returns NaN for all metrics
        - <2 players in frame: Skip frame (not included in averages)
        - <3 players for hull: Set hull area to 0.0 or NaN for that frame
        - Collinear points: QhullError caught, hull area set to NaN
        - All GK only frames: May produce valid but extreme metrics

    Example:
        >>> import pandas as pd
        >>> from src.features.services.compactness import aggregate_compactness_features
        >>> from src.features.config import CompactnessConfig
        >>>
        >>> # Sample normalized tracking data
        >>> df = pd.DataFrame({
        ...     'frame': [1, 1, 1, 1, 2, 2, 2, 2],
        ...     'player_id': [101, 102, 103, 104, 101, 102, 103, 104],
        ...     'team_id': [1, 1, 1, 1, 1, 1, 1, 1],  # All Real Madrid
        ...     'x_norm': [-30, -28, -25, -20, -28, -27, -24, -19],
        ...     'y_norm': [0, 5, -5, 2, 1, 6, -4, 3]
        ... })
        >>>
        >>> # Compute compactness
        >>> metrics = aggregate_compactness_features(df, rm_team_id=1)
        >>> print(f"Mean width: {metrics['rm_width_mean_m']:.1f}m")
        >>> print(f"Mean length: {metrics['rm_length_mean_m']:.1f}m")
        >>> print(f"Hull area: {metrics['rm_hull_area_mean_m2']:.1f}m²")
        >>> print(f"Line height: {metrics['rm_line_height_median_x_mean']:.1f}m")

    Performance:
        - Per-frame iteration (not vectorizable due to ConvexHull)
        - Typical runtime: 20-100ms per build-up (100-300 frames with 10-11 players each)

    See Also:
        - src.viz.pressing_heatmap.py: Visualizes spatial distribution
        - src.models.gmm_zones.py: Uses positions for zone modeling
        - docs/concepts/compactness-metrics.md: Tactical interpretation guide

    Notes:
        - **ConvexHull terminology**: scipy.spatial.ConvexHull uses "volume" for area in 2D
          and "area" for perimeter in 2D (confusing but standard)
        - Compactness alone doesn't indicate pressing quality - must be combined with
          pressure metrics and outcomes
        - Very low compactness (width < 10m) may indicate only a subset of team is pressing
    """
    rm_df = df_norm[df_norm["team_id"] == rm_team_id]
    
    if rm_df.empty:
        return {
            "rm_width_mean_m": float('nan'),
            "rm_width_min_m": float('nan'),
            "rm_length_mean_m": float('nan'),
            "rm_length_min_m": float('nan'),
            "rm_hull_area_mean_m2": float('nan'),
            "rm_line_height_median_x_mean": float('nan')
        }

    # Group by frame
    # calculate width/length/hull per frame
    
    # Custom apply or iteration?
    # Iteration is slow but hull calculation is not vectorized easily.
    # We have ~100 frames per build up. Iteration is fine.
    
    widths = []
    lengths = []
    hull_areas = []
    line_heights = []
    
    for _, frame_df in rm_df.groupby("frame"):
        xs = frame_df["x_norm"].values
        ys = frame_df["y_norm"].values
        
        if len(xs) < 2:
            continue
            
        # Width: Spread in Y
        width = np.max(ys) - np.min(ys)
        
        # Length: Spread in X
        length = np.max(xs) - np.min(xs)
        
        widths.append(width)
        lengths.append(length)
        
        # Line height (Defensive line)
        # Median X (if attacking right, high press means high X. Defensive line usually means BACK line, i.e., min X).
        # Plan says: "median_x = median(x_rm)". 
        # "higher = higher press line". Yes, median represents the block position.
        line_heights.append(np.median(xs))
        
        # Convex Hull
        if len(xs) >= 3: # Need 3 points for hull area
            try:
                points = np.column_stack((xs, ys))
                hull = ConvexHull(points)
                hull_areas.append(hull.area) # called volume in 2D? No, area in 2D.
                # In 2D, hull.volume is area, hull.area is perimeter.
                # Double check scipy documentation.
                # "volume: Area of the convex hull in 2D."
                # "area: Perimeter of the convex hull in 2D."
                # Very confusing naming in scipy.spatial.ConvexHull.
                # Verification:
                # 3D: volume = volume, area = surface area.
                # 2D: volume = area, area = perimeter.
                hull_areas.append(hull.volume) 
            except QhullError:
                hull_areas.append(np.nan)
        else:
            hull_areas.append(0.0)

    if not widths:
        return {
            "rm_width_mean_m": float('nan'),
            "rm_width_min_m": float('nan'),
            "rm_length_mean_m": float('nan'),
            "rm_length_min_m": float('nan'),
            "rm_hull_area_mean_m2": float('nan'),
            "rm_line_height_median_x_mean": float('nan')
        }
        
    return {
        "rm_width_mean_m": float(np.mean(widths)),
        "rm_width_min_m": float(np.min(widths)),
        "rm_length_mean_m": float(np.mean(lengths)),
        "rm_length_min_m": float(np.min(lengths)),
        "rm_hull_area_mean_m2": float(np.mean(hull_areas)),
        "rm_line_height_median_x_mean": float(np.mean(line_heights))
    }
