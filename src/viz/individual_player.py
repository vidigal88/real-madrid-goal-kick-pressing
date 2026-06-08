import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
from src.viz.plots import draw_pitch
import matplotlib.patheffects as path_effects
from scipy.spatial import ConvexHull

# MATCHING STYLES FROM cluster_comparison.py
BG_COLOR = "#FAF9F4"
PITCH_LINE_COLOR = "#999999"
TEXT_COLOR = "#111111"

# RM Colors
RM_COLOR = "#d20515"  # Red
OPP_COLOR = "#4A90E2"  # Blue

def plot_individual_player_movement(
    player_id,
    player_name,
    player_number,
    temporal_positions,  # {t: (x, y)} for t in [0, 2, 4, 6, 8, 10]
    temporal_ball,  # {t: (x, y)}
    temporal_opp_avg,  # {t: {pid: (x, y)}} - opponent positions
    role,  # 'trigger', 'support', or 'blocker'
    title,
    out_path
):
    """
    Plots individual player movement over time with directional arrows.

    Args:
        player_id: Player ID
        player_name: Player name for display
        player_number: Player jersey number
        temporal_positions: Dict mapping timestep to (x, y) position
        temporal_ball: Dict mapping timestep to ball (x, y) position
        temporal_opp_avg: Dict mapping timestep to opponent positions {pid: (x, y)}
        role: Player role ('trigger', 'support', or 'blocker')
        title: Plot title
        out_path: Output file path

    Returns:
        None (saves figure to out_path)
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw pitch
    draw_pitch(ax, color=BG_COLOR, line_color=PITCH_LINE_COLOR)

    # Attack direction indicator
    ax.add_patch(FancyArrowPatch((-15, -38), (15, -38), arrowstyle='->', mutation_scale=20,
                                 color='#555555', lw=2, zorder=5))
    ax.text(0, -40, "Attack Direction", ha='center', va='top', fontsize=12, color='#555555', style='italic')

    # Define timestamps
    timestamps = sorted(temporal_positions.keys())

    # Draw opponent formations (faded) at each timestep
    for t in timestamps:
        if t in temporal_opp_avg and temporal_opp_avg[t]:
            opp_positions = list(temporal_opp_avg[t].values())
            if len(opp_positions) >= 3:
                opp_pts = np.array(opp_positions)
                try:
                    # Draw convex hull (very faded)
                    hull = ConvexHull(opp_pts)
                    poly = Polygon(opp_pts[hull.vertices], facecolor=OPP_COLOR, edgecolor=OPP_COLOR,
                                   alpha=0.05, lw=0.5, zorder=1)
                    ax.add_patch(poly)
                except:
                    pass

                # Draw opponent dots (small, faded)
                for pos in opp_positions:
                    ax.add_patch(Circle(pos, radius=0.7, facecolor=OPP_COLOR, edgecolor='none',
                                        alpha=0.2, zorder=2))

    # Draw ball positions (small black dots)
    for t in timestamps:
        if t in temporal_ball and temporal_ball[t] is not None:
            bx, by = temporal_ball[t]
            ax.scatter(bx, by, s=60, color='black', edgecolors='white', linewidth=1, zorder=5, alpha=0.5)

    # Define arrow colors (gradient from light to dark red)
    arrow_colors = ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#d20515']  # Light to dark red

    # Draw player movement arrows
    for i in range(len(timestamps) - 1):
        t_start = timestamps[i]
        t_end = timestamps[i + 1]

        if t_start in temporal_positions and t_end in temporal_positions:
            pos_start = np.array(temporal_positions[t_start])
            pos_end = np.array(temporal_positions[t_end])

            # Calculate velocity for arrow width scaling
            distance = np.linalg.norm(pos_end - pos_start)
            time_delta = t_end - t_start
            velocity = distance / time_delta if time_delta > 0 else 0.0

            # Arrow width scales with velocity
            arrow_width = min(3.0, 1.0 + velocity * 0.25)

            # Draw arrow
            arrow = FancyArrowPatch(
                tuple(pos_start), tuple(pos_end),
                arrowstyle='->', mutation_scale=20,
                color=arrow_colors[i], lw=arrow_width, zorder=10,
                path_effects=[path_effects.withStroke(linewidth=arrow_width+1, foreground='white')]
            )
            ax.add_patch(arrow)

    # Draw player position dots (color gradient)
    position_colors = ['#ffaaaa', '#ff8888', '#ff5555', '#ff2222', '#d20515', '#b00000']  # Light to dark

    for i, t in enumerate(timestamps):
        if t in temporal_positions:
            x, y = temporal_positions[t]

            # Position dot size increases over time
            dot_size = 1.5 + (i * 0.2)

            # Draw position
            ax.add_patch(Circle((x, y), radius=dot_size, facecolor=position_colors[i],
                                edgecolor='white', lw=2, alpha=0.9, zorder=11))

            # Add time label above dot
            ax.text(x, y + 2.5, f"t={int(t)}s", ha='center', va='bottom', fontsize=9,
                    color=TEXT_COLOR, weight='bold', zorder=12,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])

    # Highlight role-specific marker on final position
    if timestamps:
        final_t = timestamps[-1]
        final_x, final_y = temporal_positions[final_t]

        if role == 'trigger':
            # Black ring
            ax.add_patch(Circle((final_x, final_y), radius=3.5, facecolor='none',
                                edgecolor='black', lw=3, zorder=13))
        elif role == 'support':
            # Dark red dashed ring
            ax.add_patch(Circle((final_x, final_y), radius=3.5, facecolor='none',
                                edgecolor='#8B0000', lw=3, linestyle='--', zorder=13))
        elif role == 'blocker':
            # Grey square
            from matplotlib.patches import Rectangle
            sq_size = 5.0
            rect = Rectangle((final_x - sq_size/2, final_y - sq_size/2), sq_size, sq_size,
                             facecolor='none', edgecolor='#333333', lw=3, zorder=13)
            ax.add_patch(rect)

    # Player name and number (bottom left)
    role_label = role.capitalize() if role else "Player"
    player_info = f"#{player_number} {player_name}\n{role_label}"
    ax.text(-50, -32, player_info, fontsize=16, weight='bold', color=RM_COLOR, ha='left', va='top',
            zorder=15, path_effects=[path_effects.withStroke(linewidth=3, foreground=BG_COLOR)])

    # Set limits and aspect
    ax.set_ylim(-45, 38)
    ax.set_xlim(-60, 60)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- ADD LEGEND ---
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle as LegendRect

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d20515',
               markersize=10, label='Player Position', markeredgecolor='white'),
        Line2D([0], [0], color='#ff6666', linewidth=2, label='Movement Arrow'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OPP_COLOR,
               markersize=8, label='Opponent (Faded)', markeredgecolor='none', alpha=0.3),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=7, label='Ball Position', markeredgecolor='white', alpha=0.5)
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              frameon=True, fancybox=True, shadow=True,
              facecolor=BG_COLOR, edgecolor='#999999')

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
    fig.suptitle(title, fontsize=24, weight='bold', y=0.935, color=TEXT_COLOR, ha='left', x=0.095)
    fig.text(0.095, 0.89, "Individual Movement Pattern: Defensive Transition",
             ha='left', fontsize=18, color='#555555')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
