"""Pitch drawing and basic plotting utilities for football visualization.

This module provides foundational plotting functions for drawing football pitches
and overlaying data visualizations. All visualizations use standard pitch dimensions
and coordinate systems compatible with SkillCorner tracking data.

**Pitch Coordinate System:**

- **Center**: (0, 0) at pitch center
- **X-axis** (length): -52.5m (left goal) to +52.5m (right goal)
- **Y-axis** (width): -34m (bottom touchline) to +34m (top touchline)
- **Dimensions**: 105m × 68m (standard FIFA dimensions)

**Key Functions:**

- `draw_pitch()`: Draw complete pitch outline with markings (center circle, boxes, goals)
- Shared utility for all visualizations (heatmaps, networks, trajectories)

**Styling:**

- Customizable colors (pitch, lines, background)
- Aspect ratio preserved (equal scaling)
- Matplotlib patches for pitch elements (Rectangle, Arc, Circle)

**Usage:**

    import matplotlib.pyplot as plt
    from src.viz.plots import draw_pitch

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, color='#FAF9F4', line_color='#999999')

    # Overlay data visualization
    ax.scatter(player_x, player_y, c='red', s=100, zorder=5)

    plt.show()

**See Also:**

- src/viz/pressing_heatmap.py: Uses draw_pitch for heatmap base
- src/viz/network.py: Uses draw_pitch for network overlay
- src/viz/pressing_network.py: Uses draw_pitch for app visualizations
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, ConnectionPatch, Ellipse
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Any

# Pitch dims (Standard 105x68 or normalized -52.5 to 52.5)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def draw_pitch(ax: Optional[Axes] = None, color: str = 'white',
               line_color: str = 'black') -> Axes:
    """Draw a football pitch with standard markings.

    Args:
        ax: Matplotlib axes object. If None, uses current axes.
        color: Pitch background color. Default 'white'.
        line_color: Color for pitch markings and lines. Default 'black'.

    Returns:
        Matplotlib axes object with pitch drawn.

    Example:
        >>> fig, ax = plt.subplots(figsize=(12, 8))
        >>> draw_pitch(ax, color='#FAF9F4', line_color='#999999')
        >>> ax.scatter(x_positions, y_positions, c='red')
        >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()
        
    x_min, x_max = -PITCH_LENGTH/2, PITCH_LENGTH/2
    y_min, y_max = -PITCH_WIDTH/2, PITCH_WIDTH/2
    
    ax.add_patch(Rectangle((x_min, y_min), PITCH_LENGTH, PITCH_WIDTH, 
                          edgecolor=line_color, facecolor=color, zorder=0))
    ax.plot([0, 0], [y_min, y_max], color=line_color, linewidth=2, zorder=1)
    ax.add_patch(Arc((0, 0), 18.3, 18.3, theta1=0, theta2=360, color=line_color, zorder=1))
    ax.add_patch(Rectangle((x_min, -20.15), 16.5, 40.3, fill=False, edgecolor=line_color, zorder=1))
    ax.add_patch(Rectangle((x_max - 16.5, -20.15), 16.5, 40.3, fill=False, edgecolor=line_color, zorder=1))
    ax.add_patch(Rectangle((x_min, -9.15), 5.5, 18.3, fill=False, edgecolor=line_color, zorder=1))
    ax.add_patch(Rectangle((x_max - 5.5, -9.15), 5.5, 18.3, fill=False, edgecolor=line_color, zorder=1))
    
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.axis("off")
    return ax



def plot_cluster_summary(features_df: pd.DataFrame, clusters_df: pd.DataFrame,
                         out_path: Path) -> None:
    """Generate box plots comparing feature distributions across clusters.

    Args:
        features_df: DataFrame with feature columns and build_up_id.
        clusters_df: DataFrame with cluster_id and build_up_id columns.
        out_path: Output file path for the saved plot.

    Returns:
        None. Saves plot to out_path.

    Example:
        >>> features = pd.read_parquet("features.parquet")
        >>> clusters = pd.read_parquet("clusters.parquet")
        >>> plot_cluster_summary(features, clusters, Path("cluster_summary.png"))
    """
    merged = features_df.merge(clusters_df, on="build_up_id", how="inner")
    cols = ["pressure_frames_ratio", "pressure_bursts_n", 
            "rm_width_mean_m", "rm_line_height_median_x_mean",
            "mean_closing_speed_mps"]
            
    # Filter only overlapping cols
    cols = [c for c in cols if c in merged.columns]
            
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x="cluster_id", y=col, data=merged)
        plt.title(col)
        
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_topic_heatmap(W: pd.DataFrame, out_path: Path) -> None:
    """Generate heatmap visualization of NMF topic-token weight matrix.

    Args:
        W: Token-topic weights matrix (tokens × topics).
        out_path: Output file path for the saved plot.

    Returns:
        None. Saves plot to out_path.

    Example:
        >>> W = pd.read_parquet("W.parquet")
        >>> plot_topic_heatmap(W, Path("topic_heatmap.png"))
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(W, cmap="viridis")
    plt.title("Topic Weights per Build-Up (W Matrix)")
    plt.savefig(out_path)
    plt.close()


