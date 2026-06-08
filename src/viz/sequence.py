import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, Polygon, Circle
from src.viz.plots import draw_pitch
import matplotlib.patheffects as path_effects
from scipy.spatial import ConvexHull

# MATCHING STYLES FROM cluster_comparison.py
BG_COLOR = "#FAF9F4"
PITCH_LINE_COLOR = "#999999"
TEXT_COLOR = "#111111"

# RM Colors (Using Red/Blue scheme from comparison)
RM_COLOR = "#d20515"       # Red
OPP_COLOR = "#4A90E2"      # Blue

def plot_temporal_sequence(
    temporal_data, # dict: {t: {'rm': {pid: (x,y)}, 'opp': {pid: (x,y)}, 'ball': (x,y), ...}}
    rm_player_names,
    rm_player_numbers,
    title,
    out_path,
    trigger_pid=None,
    support_pids=None,
    blocker_pid=None
):
    """
    Plots a multi-panel grid (2 rows x 3 cols) showing the evolution of the cluster
    at different time steps (e.g. t=2, 4, 6, 8, 10).
    Matches styles of cluster_comparison.py.
    """
    timestamps = sorted(temporal_data.keys())
    n_plots = len(timestamps)
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.set_facecolor(BG_COLOR)
    # Tighter margins to maximize pitch size
    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02, hspace=0.1)
    axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
        
    for idx, t in enumerate(timestamps):
        ax = axes[idx]
        data = temporal_data[t]
        
        ax.set_facecolor(BG_COLOR)
        draw_pitch(ax, color=BG_COLOR, line_color=PITCH_LINE_COLOR)
        
        # 0. ATTACK DIRECTION (Match cluster_comparison)
        ax.add_patch(FancyArrowPatch((-15, -38), (15, -38), arrowstyle='->', mutation_scale=20, 
                                     color='#555555', lw=2, zorder=5))
        ax.text(0, -40, "Attack Direction", ha='center', va='top', fontsize=10, color='#555555', style='italic')

        # --- DRAW OPPONENT (Blue Hull + Circles) ---
        if 'opp' in data and data['opp']:
            pts = np.array(list(data['opp'].values()))
            # Hull
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    OPP_HULL_COLOR = "#4A90E2"
                    poly = Polygon(pts[hull.vertices], facecolor=OPP_HULL_COLOR, edgecolor=OPP_COLOR,
                                  alpha=0.1, lw=1, zorder=1)
                    ax.add_patch(poly)
                except: pass
            
            # Points
            for pid, (x, y) in data['opp'].items():
                ax.add_patch(Circle((x, y), radius=1.0, facecolor=OPP_COLOR, edgecolor='white', alpha=0.9, zorder=2))

        # --- DRAW RM (Red Hull + Circles + Labels) ---
        if 'rm' in data and data['rm']:
            pts = np.array(list(data['rm'].values()))
            RM_HULL_COLOR = "#d20515"
            
            # Hull
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    poly = Polygon(pts[hull.vertices], facecolor=RM_HULL_COLOR, edgecolor=RM_COLOR,
                                  alpha=0.2, lw=2, linestyle='-', zorder=1)
                    ax.add_patch(poly)
                except: pass

            # Points
            for pid, (x, y) in data['rm'].items():
                ax.add_patch(Circle((x, y), radius=1.5, facecolor=RM_COLOR, edgecolor='white', alpha=1.0, zorder=3))
                
                # Number
                num = rm_player_numbers.get(pid, 0)
                if num > 0:
                    ax.text(x, y, str(num), ha='center', va='center', color='white', 
                            fontsize=9, weight='bold', zorder=5)
                    
                # Name (Above)
                name = rm_player_names.get(pid, "")
                if name:
                    txt = ax.text(x, y+2, name, ha='center', va='bottom', fontsize=8, 
                                  color=TEXT_COLOR, weight='bold', zorder=5)
                    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])
                
        # 3. Current ball position
        ball = data.get('ball')
        if ball is not None:
            try:
                bx, by = np.asarray(ball, dtype=float)[:2]
                if np.isfinite(bx) and np.isfinite(by):
                    ax.scatter(bx, by, s=130, color='black', edgecolors='white', linewidth=1.8, zorder=8)
            except Exception:
                pass
            
        # Title of Subplot
        panel_title = "Goal-kick setup" if t == 0.0 else f"Kick + {t:.0f}s"
        ax.set_title(panel_title, fontsize=18, weight='bold', color=TEXT_COLOR, pad=10)

        # Effective sample size for this timestamp. Later panels can use fewer
        # build-ups when the saved tracking window ends before the requested time.
        n_build_ups = data.get('n_build_ups')
        total_build_ups = data.get('total_build_ups')
        if n_build_ups is not None:
            count_label = (
                f"Goal kicks: {int(n_build_ups)}/{int(total_build_ups)}"
                if total_build_ups
                else f"Goal kicks: {int(n_build_ups)}"
            )
            ax.text(
                -56,
                34,
                count_label,
                ha='left',
                va='top',
                fontsize=9,
                color=TEXT_COLOR,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor=BG_COLOR,
                    edgecolor='#999999',
                    alpha=0.9,
                ),
                zorder=9,
            )
        
        # Limits (Match cluster_comparison)
        ax.set_ylim(-45, 38)
        ax.set_xlim(-60, 60)
        ax.set_aspect('equal')
        ax.axis('off')
        
    # --- GLOBAL HEADER ---
    # Logo (Matched Coordinates: 0.08, 0.88)
    logo_path = "data/assets/rm_logo.png"
    try:
        # Top Left (x=0.08)
        logo_ax = fig.add_axes([0.01, 0.88, 0.08, 0.08]) 
        logo_ax.axis('off')
        img = Image.open(logo_path)
        logo_ax.imshow(img)
    except: pass
    
    fig.suptitle(title, fontsize=28, weight='bold', y=0.935, x=0.095, ha='left', color=TEXT_COLOR)
    fig.text(0.095, 0.89, "Temporal Evolution: Defensive Structure Adjustment",
             ha='left', fontsize=20, color='#555555')
    fig.text(
        0.095,
        0.865,
        "Player and ball positions are cluster averages at each timestamp.",
        ha='left',
        fontsize=12,
        color='#666666',
    )

    # --- ADD LEGEND ---
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OPP_COLOR,
               markersize=10, label='Opponent', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RM_COLOR,
               markersize=10, label='Real Madrid', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=8, label='Ball', markeredgecolor='white')
    ]

    # Add legend to figure (top right)
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.85), fontsize=11,
               frameon=True, fancybox=True, shadow=True,
               facecolor=BG_COLOR, edgecolor='#999999')

    plt.tight_layout(rect=[0, 0, 1, 0.90]) # Adjust bottom slightly
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
