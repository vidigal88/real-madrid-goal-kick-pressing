import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Set, List, Optional
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import matplotlib.patheffects as path_effects
from scipy.spatial import ConvexHull
from src.viz.plots import draw_pitch

def calculate_hull_area(points: np.ndarray) -> float:
    try:
        if len(points) < 3: return 0.0
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume attribute is the Area
    except:
        return 0.0

def calculate_centroid(points: np.ndarray) -> np.ndarray:
    if len(points) == 0: return np.array([0.0, 0.0])
    return np.mean(points, axis=0)

def plot_cluster_comparison(
    initial_opp: Dict[int, Tuple[float, float]],
    initial_rm: Dict[int, Tuple[float, float]],
    initial_ball: Tuple[float, float],
    target_opp: Dict[int, Tuple[float, float]],
    target_rm: Dict[int, Tuple[float, float]],
    target_ball: Tuple[float, float],
    rm_player_names: Dict[int, str],
    rm_player_numbers: Dict[int, int],
    title: str,
    out_path: str,
    trigger_pid: Optional[int] = None,
    support_pids: Optional[List[int]] = None,
    blocker_pid: Optional[int] = None
):
    """
    Match-Analyst Ready Comparison: Initial (t=0) vs Target (t=10s).
    Features: Ghost Overlays, Centroids, Shift Vectors, Area Metrics.
    """
    
    # Colors & Styles
    BG_COLOR = "#FAF9F4"
    PITCH_LINE_COLOR = "#999999"
    OPP_COLOR = "#4A90E2"      # Blue
    OPP_HULL_COLOR = "#4A90E2"
    RM_COLOR = "#d20515"       # Red
    RM_HULL_COLOR = "#d20515"
    GHOST_COLOR = "#FFAAAA"    # Faded Red
    BALL_COLOR = "#000000"
    DROP_ZONE_COLOR = "#FFD700" # Gold
    TEXT_COLOR = "#111111"
    
    # Setup Figure: 1x2 Subplots (Wider/Taller for max pitch visibility)
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    fig.set_facecolor(BG_COLOR)
    # Tight layout margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.02)
    
    # Titles
    # Titles (Left Aligned, Tighter Padding)
    # Titles (Centered, Closer to Pitch via ylim adjust)
    axes[0].set_title("Initial State (t=0) | Trigger", fontsize=20, weight='bold', color=TEXT_COLOR, pad=10, loc='center')
    axes[1].set_title("Target Response (t=10s) | Block Shift", fontsize=20, weight='bold', color=TEXT_COLOR, pad=10, loc='center')

    # Convert Dicts to Arrays for Calculations
    init_rm_pts = np.array(list(initial_rm.values())) if initial_rm else np.empty((0,2))
    targ_rm_pts = np.array(list(target_rm.values())) if target_rm else np.empty((0,2))
    
    init_centroid = calculate_centroid(init_rm_pts)
    targ_centroid = calculate_centroid(targ_rm_pts)
    
    init_area = calculate_hull_area(init_rm_pts)
    targ_area = calculate_hull_area(targ_rm_pts)
    
    # --- DRAW FUNCTION ---
    def draw_state(ax, opp_pos, rm_pos, ball_pos, is_target_ax=False):
        ax.set_facecolor(BG_COLOR)
        draw_pitch(ax, color=BG_COLOR, line_color=PITCH_LINE_COLOR)
        
        # 1. ATTACK DIRECTION (Bottom Center)
        # Centered at x=0, Below pitch (y=-34). Moved closr to -38 (was -45).
        ax.add_patch(FancyArrowPatch((-15, -38), (15, -38), arrowstyle='->', mutation_scale=20, 
                                     color='#555555', lw=2, zorder=5))
        # Text moved up to -40 (was -44) to be very close to arrow (-38)
        ax.text(0, -40, "Attack Direction", ha='center', va='top', fontsize=12, color='#555555', style='italic')

        # 2. GHOST OVERLAY (Only on Target Axis)
        if is_target_ax and len(init_rm_pts) >= 3:
            try:
                hull = ConvexHull(init_rm_pts)
                # Ghost Hull (Dashed, Faded)
                poly = Polygon(init_rm_pts[hull.vertices], facecolor="none", edgecolor=GHOST_COLOR,
                              alpha=0.6, lw=2, linestyle='--', zorder=0)
                ax.add_patch(poly)
            except: pass

        # 3. OPPONENT (Blue)
        if opp_pos and len(opp_pos) > 0:
            pts = np.array(list(opp_pos.values()))
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    poly = Polygon(pts[hull.vertices], facecolor=OPP_HULL_COLOR, edgecolor=OPP_COLOR,
                                  alpha=0.1, lw=1, zorder=1)
                    ax.add_patch(poly)
                except: pass
            for pid, (x, y) in opp_pos.items():
                ax.add_patch(Circle((x, y), radius=1.0, facecolor=OPP_COLOR, edgecolor='white', alpha=0.9, zorder=2))

        # 4. REAL MADRID (Red)
        if rm_pos:
            pts = np.array(list(rm_pos.values()))
            current_area = 0.0
            
            # Hull
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    current_area = hull.volume
                    poly = Polygon(pts[hull.vertices], facecolor=RM_HULL_COLOR, edgecolor=RM_COLOR,
                                  alpha=0.2, lw=2, linestyle='-', zorder=1)
                    ax.add_patch(poly)
                except: pass
            
            # Area Label (Top Left Corner, Inside Pitch)
            area_txt = f"Area: {current_area:.0f} m²"
            # Moved up to 32 (was 30)
            ax.text(-50, 32, area_txt, fontsize=14, weight='bold', color=RM_COLOR, ha='left', va='top', zorder=6,
                   path_effects=[path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])

            # Centroid
            centroid = calculate_centroid(pts)
            ax.scatter(centroid[0], centroid[1], marker='+', s=150, color='black', zorder=4, linewidths=2)
            
            # Shift Vector (Only on Target Axis)
            if is_target_ax:
                # Arrow from Init Centroid -> Targ Centroid
                arrow = FancyArrowPatch((init_centroid[0], init_centroid[1]), (targ_centroid[0], targ_centroid[1]),
                                       arrowstyle='->', mutation_scale=20, color='black', lw=2, linestyle='--', zorder=4)
                ax.add_patch(arrow)
                
                # Delta Area Label (Below Area Label)
                delta = targ_area - init_area
                sign = "+" if delta > 0 else ""
                # Moved up to 28 (was 28... wait, Area is 32. So Change should be 28 or 29?)
                # If Area is 32 (top), it takes ~2 units height? Font 14.
                # put Change at 28. (Gap of 4 units).
                ax.text(-50, 28, f"Change: {sign}{delta:.0f} m²", fontsize=12, color=TEXT_COLOR, ha='left', va='top', zorder=6,
                       path_effects=[path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])

            # Dots & Labels
            for pid, (x, y) in rm_pos.items():
                ax.add_patch(Circle((x, y), radius=1.5, facecolor=RM_COLOR, edgecolor='white', alpha=1.0, zorder=3))

                # --- ROLE IDENTIFICATIONS (Show on BOTH panels) ---
                # Trigger (First Presser) -> Black Ring
                if trigger_pid is not None and pid == trigger_pid:
                    ax.add_patch(Circle((x, y), radius=2.5, facecolor='none', edgecolor='black', lw=3, zorder=6))

                # Support (2nd/3rd) -> Dark Red Ring
                if support_pids is not None and pid in support_pids:
                    ax.add_patch(Circle((x, y), radius=2.5, facecolor='none', edgecolor='#8B0000', lw=2.5, zorder=6, linestyle='--'))

                # Channel Blocker (Show on BOTH panels for consistency)
                if blocker_pid is not None and pid == blocker_pid:
                    # Dark Grey Square
                    from matplotlib.patches import Rectangle
                    # Centered square size 4x4
                    sq_size = 4.0
                    rect = Rectangle((x - sq_size/2, y - sq_size/2), sq_size, sq_size,
                                     facecolor='none', edgecolor='#333333', lw=3, zorder=6, linestyle='-')
                    ax.add_patch(rect)

                # Player Names (Only on Target Axis)
                if is_target_ax:
                    # Smart Labeling: Offset based on position relative to centroid
                    dx = x - centroid[0]
                    dy = y - centroid[1]
                    dist = np.hypot(dx, dy)
                    if dist == 0: dist = 1

                    offset_x = (dx / dist) * 3.0
                    offset_y = (dy / dist) * 3.0

                    name = rm_player_names.get(pid, "")
                    if name:
                        txt = ax.text(x + offset_x, y + offset_y, name, fontsize=9, ha='center', va='center',
                                      color=TEXT_COLOR, weight='bold', zorder=5)
                        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])

        # 5. BALL & DROP ZONE
        if ball_pos is not None:
            try:
                bx, by = np.asarray(ball_pos, dtype=float)[:2]
            except Exception:
                bx, by = np.nan, np.nan

            if np.isfinite(bx) and np.isfinite(by):
                if is_target_ax:
                    # Draw Drop Zone Marker (Yellow X)
                    ax.scatter(bx, by, marker='X', s=200, color=DROP_ZONE_COLOR, edgecolors='black', linewidths=1.5, zorder=6, label='Ball Drop')
                    ax.text(bx, by+3, "Ball Drop", fontsize=10, weight='bold', color=TEXT_COLOR, ha='center',
                            path_effects=[path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])
                else:
                    # Normal Ball
                    ax.add_patch(Circle((bx, by), radius=0.8, facecolor=BALL_COLOR, edgecolor='white', lw=1, zorder=5))

        # Adjusted Y-limits:
        # Bottom (-75) to increase footer space (move pitch up relative to frame)
        # Top (38) to bring Title closer (Pitch top is 34. 38 is very tight)
        ax.set_ylim(-45, 38) 
        ax.set_xlim(-60, 60)
        ax.set_aspect('equal')
        ax.axis('off')

    # --- RENDER PANELS ---
    draw_state(axes[0], initial_opp, initial_rm, initial_ball, is_target_ax=False)
    draw_state(axes[1], target_opp, target_rm, target_ball, is_target_ax=True)
    
    # --- HEADER & LOGO ---
    logo_path = "data/assets/rm_logo.png"
    logo_path = "data/assets/rm_logo.png"
    try:
        # Moved Logo "more to the left" to 0.08 (was 0.125)
        # User requested visual push leftwards.
        logo_ax = fig.add_axes([0.01, 0.88, 0.08, 0.08])
        logo_ax.axis('off')
        img = Image.open(logo_path)
        logo_ax.imshow(img)
    except: pass
    
    # New Title Logic
    fig.suptitle(title.replace("Pressure Network", "Defensive Block Transition"), 
                 fontsize=26, weight='bold', y=0.935, color=TEXT_COLOR, ha='left', x=0.095)
    fig.text(0.095, 0.89, "Tactical Analysis: Centroid Shift & Space Compactness", 
             ha='left', fontsize=18, color='#555555')

    # ATTACK DIRECTION (Removed Figure-Level, moved back to Axis-Level)

    # --- ADD LEGEND ---
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle as LegendRect

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OPP_COLOR,
               markersize=10, label='Opponent', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RM_COLOR,
               markersize=10, label='Real Madrid', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=12, markeredgecolor='black', markeredgewidth=2, label='Trigger (First Presser)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=12, markeredgecolor='#8B0000', markeredgewidth=2, linestyle='--', label='Support Pressers'),
        LegendRect((0, 0), 1, 1, facecolor='none', edgecolor='#333333', linewidth=2, label='Channel Blocker'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BALL_COLOR,
               markersize=8, label='Ball', markeredgecolor='white'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=DROP_ZONE_COLOR,
               markersize=10, label='Ball Drop Zone', markeredgecolor='black'),
        Line2D([0], [0], color=GHOST_COLOR, linestyle='--', linewidth=2,
               label='Initial Position (Ghost)', alpha=0.6),
        Line2D([0], [0], marker='+', color='w', markerfacecolor='black',
               markersize=12, markeredgewidth=2, label='Team Centroid')
    ]

    # Add legend to figure (top right, outside plot area)
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.85), fontsize=11,
               frameon=True, fancybox=True, shadow=True,
               facecolor=BG_COLOR, edgecolor='#999999')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
