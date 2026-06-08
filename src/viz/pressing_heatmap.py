"""Pressing heatmap visualization for spatial density analysis.

This module generates heatmap visualizations showing where Real Madrid's pressing
events occur most frequently across multiple build-ups. Heatmaps reveal spatial
patterns in pressing behavior, such as:
- Which pitch zones receive most pressing attention
- Asymmetric pressing tendencies (left vs right bias)
- High-intensity zones (frequent pressing events)
- Low-activity zones (gaps in pressing coverage)

**Pressing Event Definition:**

A pressing event occurs when a Real Madrid player:
1. Moves toward the ball carrier (reducing distance)
2. With closing velocity ≥ 1.0 m/s (approaching quickly)

This captures active pressing attempts, not passive proximity.

**Grid-Based Density:**

The pitch is divided into a 10×5 grid (50 cells):
- **X-axis**: 10 bins from left goal (-52.5m) to right goal (+52.5m), 10.5m width each
- **Y-axis**: 5 bins from bottom touchline (-34m) to top touchline (+34m), 13.6m height each

Each cell accumulates pressing event counts across all analyzed build-ups.

**Visualization:**

Heatmaps are rendered using matplotlib with:
- Background color: #FAF9F4 (off-white, matches aesthetic)
- Pitch lines: #999999 (gray)
- Heatmap colormap: "Reds" (white = low, dark red = high)
- Pitch outline overlay for spatial reference

**Usage:**

    from src.viz.pressing_heatmap import create_pressing_heatmap

    # Generate heatmap from build-up data
    fig = create_pressing_heatmap(
        all_build_ups=build_up_ids,
        loader=window_loader,
        metadata_dicts=metadata,
        title="Real Madrid Pressing Heatmap - Season 2023/24"
    )

    fig.savefig("pressing_heatmap.png", dpi=300, bbox_inches='tight')

**Applications:**

- **Tactical Analysis**: Identify pressing focus areas and coverage gaps
- **Opposition Scouting**: Show opponents where RM pressure is strongest
- **Performance Tracking**: Compare heatmaps across matches/seasons
- **Strategic Planning**: Adjust pressing zones based on density patterns

**See Also:**

- src/viz/plots.py: Pitch drawing utilities
- src/features/services/pressure.py: Pressing event detection
- docs/concepts/pressing-patterns.md: Pressing analysis methodology
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from src.viz.plots import draw_pitch
import matplotlib.patheffects as path_effects
from tqdm import tqdm

# MATCHING STYLES
BG_COLOR = "#FAF9F4"
PITCH_LINE_COLOR = "#999999"
TEXT_COLOR = "#111111"

def calculate_pressing_event_density(
    all_build_ups,
    loader,
    game_id_map,
    gk_side_map,
    opp_id_map,
    prepare_frame_data_fn,
    normalize_coordinates_fn,
    enrich_with_team_id_fn,
    infer_ball_carrier_fn,
    time_to_seconds_fn,
    ball_proximity_threshold=5.0,
    opponent_proximity_threshold=3.0
):
    """Calculate spatial density of unique pressing initiation events across build-ups.

    A pressing event is counted when a Real Madrid player STARTS pressing (unique events only).
    Pressing is defined as being simultaneously:
    1. Within ball_proximity_threshold meters of the BALL CARRIER (default: 5m)
    2. Within opponent_proximity_threshold meters of any opponent (default: 3m)

    Continuous pressing in the same grid cell by the same player is deduplicated - only
    the initiation moment is counted. This captures WHERE pressing starts, not duration.

    **Grid Structure:**
    ```
    Pitch divided into 50 cells (10 columns × 5 rows):

    Y-axis (width)
    ▲
    │  [50 cells total]
    │  Row 5 (top)    : y ∈ [20.4, 34]
    │  Row 4          : y ∈ [6.8, 20.4]
    │  Row 3 (center) : y ∈ [-6.8, 6.8]
    │  Row 2          : y ∈ [-20.4, -6.8]
    │  Row 1 (bottom) : y ∈ [-34, -20.4]
    └────────────────────────────────────> X-axis (length)
         -52.5 ← [10 columns] → +52.5
    ```

    Args:
        all_build_ups: List of build_up_id integers to analyze
        loader: WindowLoader instance for loading tracking data
        game_id_map: Dict {build_up_id: game_id} for metadata lookup
        gk_side_map: Dict {build_up_id: "left"|"right"} for normalization
        opp_id_map: Dict {build_up_id: opponent_team_id} for filtering
        prepare_frame_data_fn: Preprocessing function (handle missing data)
        normalize_coordinates_fn: Coordinate normalization function
        enrich_with_team_id_fn: Team ID enrichment function
        infer_ball_carrier_fn: Ball possession inference function
        time_to_seconds_fn: Time conversion function
        ball_proximity_threshold: Maximum distance to ball (meters) for pressing event
        opponent_proximity_threshold: Maximum distance to opponent (meters) for pressing event

    Returns:
        grid: NumPy array of shape (5, 10) containing pressing event counts.
              grid[row, col] = count of pressing events in that cell.

    Notes:
        - Pressing event: RM player near BALL CARRIER and near opponent (active engagement)
        - Deduplication: Only counts when pressing STARTS, not continuous frames
        - Ball carrier must be identified (within 3m of ball via infer_ball_carrier)
        - Normalized coordinates: Opponent always attacks right (positive X)
        - Skips build-ups with errors (missing data, failed processing)
    """
    # Grid definition: x: [-52.5, 52.5], y: [-34, 34]
    x_bins = np.linspace(-52.5, 52.5, 11)  # 10 cells in x
    y_bins = np.linspace(-34, 34, 6)  # 5 cells in y
    grid = np.zeros((5, 10))  # rows=y, cols=x

    print(f"Calculating unique pressing initiation events (carrier <={ball_proximity_threshold}m, opponent <={opponent_proximity_threshold}m)...")

    for bid in tqdm(all_build_ups, desc="Processing build-ups"):
        try:
            df = loader.load_build_up(bid)
            df = prepare_frame_data_fn(df)
            gid = game_id_map.get(bid, 0)
            df = enrich_with_team_id_fn(df, gid)
            side = gk_side_map.get(bid, "left")
            df_norm = normalize_coordinates_fn(df, side)

            opp_id = opp_id_map.get(bid)
            df_norm = infer_ball_carrier_fn(df_norm, opp_id)

            if "time_seconds" not in df_norm.columns:
                df_norm["time_seconds"] = df_norm["time"].apply(time_to_seconds_fn)

            # Get unique timestamps
            timestamps = sorted(df_norm["time_seconds"].unique())

            # Track pressing state for deduplication
            # Set of (player_id, grid_x, grid_y) tuples from previous frame
            pressing_last_frame = set()

            # For each frame, check for pressing situations
            for t in timestamps:
                frame = df_norm[df_norm["time_seconds"] == t]

                # Get ball carrier position (not just ball position)
                ball_carrier_id = frame.iloc[0]["ball_carrier_id"]
                if pd.isna(ball_carrier_id):
                    continue  # No identified ball carrier in this frame

                # Find the ball carrier's position
                carrier_row = frame[(frame["player_id"] == ball_carrier_id) &
                                   (frame["team_id"] == opp_id)]
                if carrier_row.empty:
                    continue

                carrier_pos = np.array([carrier_row.iloc[0]["x_norm"],
                                       carrier_row.iloc[0]["y_norm"]])

                # Get RM players and opponents
                rm_players = frame[(frame["team_id"] != opp_id) & (frame["is_ball"] == False)]
                opp_players = frame[(frame["team_id"] == opp_id) & (frame["is_ball"] == False)]

                if opp_players.empty:
                    continue

                # Get opponent positions as array for vectorized distance calculation
                opp_positions = opp_players[["x_norm", "y_norm"]].values

                # Track pressing events in this frame for deduplication
                pressing_this_frame = set()

                # For each RM player, check if they're pressing
                for _, rm_player in rm_players.iterrows():
                    rm_pos = np.array([rm_player["x_norm"], rm_player["y_norm"]])
                    pid = rm_player["player_id"]

                    # Check distance to ball carrier
                    dist_to_carrier = np.linalg.norm(rm_pos - carrier_pos)
                    if dist_to_carrier > ball_proximity_threshold:
                        continue  # Too far from ball carrier

                    # Check distance to nearest opponent (vectorized)
                    distances_to_opponents = np.linalg.norm(opp_positions - rm_pos, axis=1)
                    min_dist_to_opponent = distances_to_opponents.min()

                    if min_dist_to_opponent <= opponent_proximity_threshold:
                        # This is a pressing condition! RM player is near carrier AND near opponent
                        # Find grid cell
                        x_idx = np.digitize(rm_pos[0], x_bins) - 1
                        y_idx = np.digitize(rm_pos[1], y_bins) - 1

                        # Ensure within bounds
                        x_idx = np.clip(x_idx, 0, 9)
                        y_idx = np.clip(y_idx, 0, 4)

                        # Create key for this pressing instance
                        press_key = (pid, x_idx, y_idx)
                        pressing_this_frame.add(press_key)

                        # Only count if NEW pressing event (not in previous frame)
                        if press_key not in pressing_last_frame:
                            grid[y_idx, x_idx] += 1

                # Update for next frame iteration
                pressing_last_frame = pressing_this_frame

        except Exception as e:
            continue

    # Debug output
    print(f"Grid shape: {grid.shape}")
    print(f"Total unique pressing initiations detected: {grid.sum()}")
    print(f"Max initiations in single cell: {grid.max()}")
    print(f"Non-zero cells: {np.count_nonzero(grid)}/{grid.size}")
    print(f"Grid values range: [{grid.min():.2f}, {grid.max():.2f}]")

    return grid




def plot_heatmap(grid, title, out_path, colorbar_label, cmap='Reds'):
    """
    Plot heatmap on pitch with 50-cell grid overlay.

    Args:
        grid: 2D numpy array (5 rows x 10 cols)
        title: Plot title
        out_path: Output file path
        colorbar_label: Label for colorbar
        cmap: Colormap ('Reds' for pressing events, 'Blues' for position density)

    Returns:
        None (saves figure to out_path)
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw pitch
    draw_pitch(ax, color=BG_COLOR, line_color=PITCH_LINE_COLOR)

    # Grid extents
    x_min, x_max = -52.5, 52.5
    y_min, y_max = -34, 34

    # Normalize grid for better color distribution
    # Use percentile-based normalization to handle outliers
    vmax = np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1.0
    vmin = 0

    # Plot heatmap with 'nearest' interpolation to show distinct cells
    # Flip grid vertically for correct orientation (origin='lower')
    heatmap = ax.imshow(grid, extent=[x_min, x_max, y_min, y_max],
                        origin='lower', cmap=cmap, alpha=0.70, zorder=3,
                        interpolation='nearest', vmin=vmin, vmax=vmax,
                        aspect='auto')

    # Add grid lines to show cell boundaries
    x_bins = np.linspace(x_min, x_max, 11)  # 10 cells
    y_bins = np.linspace(y_min, y_max, 6)   # 5 cells

    # Draw vertical grid lines
    for x in x_bins:
        ax.plot([x, x], [y_min, y_max], color='white', linewidth=1.5, alpha=0.3, zorder=4)

    # Draw horizontal grid lines
    for y in y_bins:
        ax.plot([x_min, x_max], [y, y], color='white', linewidth=1.5, alpha=0.3, zorder=4)

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
    cbar.set_label(colorbar_label, fontsize=14, color=TEXT_COLOR, weight='bold')
    cbar.ax.tick_params(labelsize=11, colors=TEXT_COLOR)

    # Attack direction indicator
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((-15, -38), (15, -38), arrowstyle='->', mutation_scale=20,
                                 color='#555555', lw=2, zorder=5))
    ax.text(0, -40, "Attack Direction", ha='center', va='top', fontsize=12, color='#555555', style='italic')

    # Set limits and aspect
    ax.set_ylim(-45, 38)
    ax.set_xlim(-60, 60)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- HEADER & LOGO ---
    logo_path = "data/assets/rm_logo.png"
    try:
        logo_ax = fig.add_axes([0.01, 0.88, 0.08, 0.08])
        logo_ax.axis('off')
        img = Image.open(logo_path)
        logo_ax.imshow(img)
    except:
        pass

    # Title
    fig.suptitle(title, fontsize=26, weight='bold', y=0.935, color=TEXT_COLOR, ha='left', x=0.095)
    fig.text(0.095, 0.89, "Pressing Initiation: Ball Carrier Engagement Zones",
             ha='left', fontsize=20, color='#555555')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()

    print(f"Heatmap saved to {out_path}")
