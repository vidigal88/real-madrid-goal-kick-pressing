"""Pressing network visualization for co-pressing partnership analysis.

This module creates network graph visualizations showing pressing partnerships between
Real Madrid players. Networks reveal:
- **Nodes**: Players (sized by pressing frequency)
- **Edges**: Co-pressing partnerships (weighted by affinity)
- **Backbone**: Strongest partnerships (filtered network)

**Network Concepts:**

1. **Co-Pressing**: Two players press together when both are within pressure threshold
   of ball carrier simultaneously

2. **Affinity**: Partnership strength measured via Jaccard or Cosine similarity
   (see src/models/pressing_affinity.py)

3. **Backbone Extraction**: Filter to show only strongest partnerships, removing
   weak/noisy connections

**Visualization Features:**

- **Pitch Overlay**: Network drawn on pitch diagram for spatial context
- **Node Size**: Proportional to player's total pressing frames
- **Edge Width**: Proportional to partnership affinity
- **Edge Alpha**: Transparency indicates partnership strength
- **Red Shading**: Nodes colored by pressing intensity
- **Player Images**: Optional player photos at node positions

**Backbone Extraction:**

Reduces network clutter by keeping only significant edges:
- **k-per-node**: Keep top k strongest partnerships per player
- **Percentile threshold**: Only edges above 75th-85th percentile affinity
- **Guarantee connectivity**: Ensure each node has ≥1 edge

**Usage:**

    from src.viz.network import create_pressing_network

    fig = create_pressing_network(
        affinity_matrix=affinity_df,
        player_positions=positions_dict,
        player_metadata=metadata,
        style=PressureNetworkStyle()
    )

    fig.savefig("pressing_network.png", dpi=300)

**Applications:**

- **Partnership Analysis**: Identify coordinated pressing units (e.g., CB-DM pairs)
- **Tactical Insight**: Reveal pressing structure (centralized vs distributed)
- **Player Roles**: Distinguish connectors (high degree) from specialists (low degree)
- **Formation Analysis**: Network topology reflects tactical setup

**See Also:**

- src/models/pressing_affinity.py: Affinity metric calculation
- src/analysis/network_centrality.py: Centrality metrics (degree, betweenness)
- docs/concepts/network-analysis.md: Network analysis methodology
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional, List, Any

from src.viz.plots import draw_pitch
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


@dataclass(frozen=True)
class PressureNetworkStyle:
    # Canvas
    bg_color: str = "#FAF9F4"
    pitch_line_color: str = "#999999"
    text_color: str = "#000000"

    # Nodes
    use_red_shades: bool = True
    red_constant: str = "#D7191C"
    node_r_min: float = 1.2
    node_r_max_add: float = 2.4

    # Edges
    edge_color: str = "#444444"
    edge_alpha_min: float = 0.10
    edge_alpha_max: float = 0.60
    edge_w_min: float = 0.9
    edge_w_max_add: float = 4.5

    # Backbone
    backbone_k_per_node: int = 2
    backbone_global_percentile_fullteam: float = 85.0
    backbone_global_percentile_frequent: float = 75.0
    guarantee_one_edge_per_node: bool = True

    # Labels
    top_name_labels_fullteam: int = 5
    top_name_labels_frequent: int = 6

    # Micro-jitter
    jitter_eps: float = 0.6

    # Figure
    figsize: Tuple[int, int] = (16, 12)
    xlim: Tuple[float, float] = (-60, 60)
    ylim: Tuple[float, float] = (-54, 45)  # Extended bottom to -54 to accommodate legend
    dpi: int = 200


class PressureNetworkPlotter:
    """
    Pressure Network Plotter (SkillCorner coordinates assumed).

    Inputs:
      - avg_positions: pid -> (x, y)
      - co_press_counts: (pid1, pid2) -> weight (int or float)
      - node_sizes: pid -> size (int or float)
      - player_names, player_numbers: pid -> metadata
      - highlight_pids: set of pids to emphasize / keep in frequent view
      - show_all_players:
          * True: show all nodes; frequent/highlight only affects labeling priority
          * False: plot only highlight_pids (frequent players view)

    Features:
      - Red-only nodes (optionally shaded by involvement)
      - Backbone pruning (percentile + top-k per node)
      - Optional guarantee: at least one edge per node (prevents isolated nodes)
      - Rank-based edge width/alpha (improves discrimination when weights cluster)
      - Deterministic micro-jitter to reduce overlap
      - Optional global scaling for comparability (node_scale, edge_scale)
        NOTE: edge_scale is accepted for API compatibility; rank-based styling does not require it.
    """

    def __init__(self, style: PressureNetworkStyle = PressureNetworkStyle()):
        self.style = style

    # ---------------------------
    # Helpers: backbone, scaling, jitter, text color
    # ---------------------------
    @staticmethod
    def _robust_norm(values: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> Tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(values, p_low))
        vmax = float(np.percentile(values, p_high))
        if vmax <= vmin:
            vmin = float(values.min())
            vmax = float(values.max() + 1e-6)
        return vmin, vmax

    def _jitter_pos(self, pos: Tuple[float, float], pid: int) -> Tuple[float, float]:
        eps = self.style.jitter_eps
        dx = eps * ((pid * 37) % 7 - 3) / 10.0
        dy = eps * ((pid * 91) % 7 - 3) / 10.0
        return (pos[0] + dx, pos[1] + dy)

    @staticmethod
    def _get_text_color_for_background(bg_color) -> str:
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(bg_color)

        def linearize(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = [linearize(c) for c in rgb]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.5 else "white"

    def _backbone_edges_topk_percentile(
        self,
        co_press_counts: Dict[Tuple[int, int], int],
        valid_nodes: Set[int],
        k_per_node: int,
        global_percentile: float,
        guarantee_one_edge_per_node: bool
    ) -> Set[Tuple[int, int]]:
        if not co_press_counts:
            return set()

        items = [((u, v), w) for (u, v), w in co_press_counts.items() if u in valid_nodes and v in valid_nodes]
        if not items:
            return set()

        keep: Set[Tuple[int, int]] = set()

        # Guarantee: keep strongest incident edge per node (prevents isolates)
        if guarantee_one_edge_per_node:
            full_adj: Dict[int, List[Tuple[int, float]]] = {}
            for (u, v), w in items:
                full_adj.setdefault(u, []).append((v, float(w)))
                full_adj.setdefault(v, []).append((u, float(w)))

            for u, nbrs in full_adj.items():
                if not nbrs:
                    continue
                v_best, _ = max(nbrs, key=lambda x: x[1])
                keep.add((min(u, v_best), max(u, v_best)))

        # Percentile threshold + Top-K per node among thresholded
        weights = np.array([float(w) for _, w in items], dtype=float)
        thr = np.percentile(weights, global_percentile)

        adj: Dict[int, List[Tuple[int, float]]] = {}
        for (u, v), w in items:
            w = float(w)
            if w < thr:
                continue
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

        for u, nbrs in adj.items():
            nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)[:k_per_node]
            for v, _ in nbrs_sorted:
                keep.add((min(u, v), max(u, v)))

        return keep

    # ---------------------------
    # Public API
    # ---------------------------
    def plot(
        self,
        avg_positions: Dict[int, Tuple[float, float]],
        co_press_counts: Dict[Tuple[int, int], int],
        node_sizes: Dict[int, int],
        player_names: Dict[int, str],
        player_numbers: Dict[int, int],
        player_roles: Dict[int, str],  # kept for compatibility; not used
        highlight_pids: Set[int],
        title: str,
        out_path: str,
        show_all_players: bool = True,
        node_scale: Optional[Tuple[float, float]] = None,
        edge_scale: Optional[float] = None,  # accepted for signature stability; rank scaling does not need it
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        use_affinity_weights: bool = False,
        affinity_scores: Optional[Dict[Tuple[int, int], float]] = None,
        outcome_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        s = self.style

        # Adaptive settings
        backbone_percentile = s.backbone_global_percentile_fullteam if show_all_players else s.backbone_global_percentile_frequent
        top_name_labels = s.top_name_labels_fullteam if show_all_players else s.top_name_labels_frequent

        # Figure
        fig, ax = plt.subplots(figsize=s.figsize)
        fig.set_facecolor(s.bg_color)
        ax.set_facecolor(s.bg_color)
        draw_pitch(ax, color=s.bg_color, line_color=s.pitch_line_color)

        # Use provided limits or style defaults
        final_xlim = xlim if xlim is not None else s.xlim
        final_ylim = ylim if ylim is not None else s.ylim

        # Nodes included
        valid_nodes = set(avg_positions.keys())
        if not show_all_players:
            valid_nodes = valid_nodes.intersection(set(highlight_pids))

        # Node scaling
        size_vals = np.array([float(node_sizes.get(pid, 0.0)) for pid in valid_nodes], dtype=float)
        if node_scale is None:
            smin, smax = self._robust_norm(size_vals, 5.0, 95.0)
        else:
            smin, smax = node_scale
        size_norm = Normalize(vmin=smin, vmax=smax)

        cmap = plt.cm.Reds

        # Build graph
        G = nx.Graph()
        for pid in valid_nodes:
            x, y = avg_positions[pid]
            val = float(node_sizes.get(pid, 0.0))

            frac = np.sqrt(max(val - smin, 0.0) / (smax - smin + 1e-9))
            radius = s.node_r_min + s.node_r_max_add * frac

            is_highlight = pid in highlight_pids
            if s.use_red_shades:
                # push colors away from the near-white part of Reds
                t = float(size_norm(val))           # 0..1
                t = 0.30 + 0.70 * t                 # shift into [0.30, 1.0]
                t = t ** 0.65                       # gamma (optional; remove if you want linear)
                color = cmap(t)
            else:
                color = s.red_constant
            z_ord = 3 if is_highlight else 2

            G.add_node(
                pid,
                pos=(x, y),
                size=val,
                radius=radius,
                color=color,
                is_highlight=is_highlight,
                z_ord=z_ord
            )

        # Backbone edges
        keep_edges = self._backbone_edges_topk_percentile(
            co_press_counts=co_press_counts,
            valid_nodes=valid_nodes,
            k_per_node=s.backbone_k_per_node,
            global_percentile=backbone_percentile,
            guarantee_one_edge_per_node=s.guarantee_one_edge_per_node
        )

        for (u, v), w in co_press_counts.items():
            if u not in valid_nodes or v not in valid_nodes:
                continue
            key = (min(u, v), max(u, v))
            if key not in keep_edges:
                continue
            G.add_edge(u, v, weight=float(w))

        # Rank-based scaling for edges
        # Use affinity scores for ranking if available, otherwise use raw weights
        if use_affinity_weights and affinity_scores:
            edges_sorted = sorted(
                G.edges(data=True),
                key=lambda e: affinity_scores.get((min(e[0], e[1]), max(e[0], e[1])), 0.0)
            )
        else:
            edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0.0))

        n_edges = len(edges_sorted)
        rank_map: Dict[Tuple[int, int], float] = {}

        if n_edges <= 1:
            for (u, v, _) in edges_sorted:
                rank_map[(min(u, v), max(u, v))] = 1.0
        else:
            for i, (u, v, _) in enumerate(edges_sorted):
                rank_map[(min(u, v), max(u, v))] = i / (n_edges - 1)

        # Draw edges (use jittered positions for consistency)
        for (u, v, _) in G.edges(data=True):
            nw = rank_map.get((min(u, v), max(u, v)), 0.0)
            width = s.edge_w_min + s.edge_w_max_add * nw
            alpha = s.edge_alpha_min + (s.edge_alpha_max - s.edge_alpha_min) * nw

            (x1, y1) = self._jitter_pos(G.nodes[u]["pos"], u)
            (x2, y2) = self._jitter_pos(G.nodes[v]["pos"], v)

            ax.plot(
                [x1, x2], [y1, y2],
                linewidth=width,
                color=s.edge_color,
                alpha=alpha,
                zorder=1,
                linestyle="solid",
                solid_capstyle="round"
            )

        # Label policy: names for top-N by size + highlights
        sorted_by_size = sorted(valid_nodes, key=lambda pid: node_sizes.get(pid, 0), reverse=True)
        top_name_ids = set(sorted_by_size[:top_name_labels]).union(set(highlight_pids))

        # Draw nodes + texts (highlights last)
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]["z_ord"])
        for node_id, attrs in sorted_nodes:
            pos = self._jitter_pos(attrs["pos"], node_id)
            radius = attrs["radius"]
            color = attrs["color"]
            z_ord = attrs["z_ord"]

            # Determine border color based on outcome statistics
            edge_color = "black"
            edge_width = 1.6

            if outcome_stats and 'players' in outcome_stats and node_id in outcome_stats['players']:
                success_rate = outcome_stats['players'][node_id].get('success_rate', 0.0)
                edge_width = 2.0 + (success_rate * 2.0)  # 2-4 px based on success

                if success_rate >= 0.60:
                    edge_color = "#2ecc71"  # Green (high success)
                elif success_rate >= 0.40:
                    edge_color = "#f39c12"  # Orange/Yellow (medium success)
                else:
                    edge_color = "#e74c3c"  # Red (low success)

            ax.add_patch(
                Circle(
                    pos,
                    radius=radius,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=0.95,
                    zorder=z_ord
                )
            )

            num = player_numbers.get(node_id, "?")
            num_color = self._get_text_color_for_background(color)
            ax.text(
                pos[0], pos[1], str(num),
                fontsize=11,
                ha="center",
                va="center",
                color=num_color,
                weight="bold",
                zorder=z_ord + 1
            )

            if node_id in top_name_ids:
                name = player_names.get(node_id, str(node_id))
                offset_x = radius + 0.9
                offset_y = 0.35 if (node_id % 2 == 0) else -0.35

                txt = ax.text(
                    pos[0] + offset_x, pos[1] + offset_y,
                    name,
                    fontsize=10,
                    ha="left",
                    va="center",
                    color=s.text_color,
                    weight="bold",
                    zorder=z_ord + 1
                )
                txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=s.bg_color)])

        # Title + logo
        logo_x, logo_y = -52.5, 42  # Moved up from 40 to create 8-unit buffer from pitch top
        title_x = -48

        logo_path = "data/assets/rm_logo.png"
        try:
            logo = Image.open(logo_path)
            imagebox = OffsetImage(logo, zoom=0.15)
            ab = AnnotationBbox(imagebox, (logo_x, logo_y), frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)
            title_x = logo_x + 12.0
        except Exception:
            title_x = logo_x

        ax.text(title_x, logo_y, title, fontsize=20, weight="bold", color=s.text_color, ha="left", va="center")
        ax.text(title_x, logo_y - 3.5, "Real Madrid Pressing Network | 23/24", fontsize=12, color="#555555", ha="left", va="center")

        # Legend (centered at current x-center, fixed margin from bottom)
        # We stick to data coordinates for standard layout
        legend_y = -48  # Moved down for better spacing
        ax.text(0, legend_y + 8, "NETWORK METRICS", ha="center", fontsize=11, weight="bold", color="#222222")

        # Node involvement legend (left of center)
        legend_left_x = -30
        ax.text(legend_left_x, legend_y + 5, "Pressure Involvement", ha="center", fontsize=9, weight="bold", color="#444444")

        sample_fracs = [0.25, 0.6, 1.0]
        labels = ["Low", "Med", "High"]
        start_x = legend_left_x - 10

        cmap_legend = plt.cm.Reds
        for i, (f, lab) in enumerate(zip(sample_fracs, labels)):
            r = s.node_r_min + s.node_r_max_add * f
            cx = start_x + i * 10
            
            if s.use_red_shades:
                t_val = 0.30 + 0.70 * f
                t_val = t_val ** 0.65
                col = cmap_legend(t_val)
            else:
                col = s.red_constant

            ax.add_patch(Circle((cx, legend_y), radius=r, facecolor=col, edgecolor="black", linewidth=1.2, alpha=0.95))
            ax.text(cx, legend_y - 4.2, lab, ha="center", va="top", fontsize=8, color="#666666")

        # Separator
        ax.plot([0, 0], [legend_y - 6, legend_y + 6], color="#CCCCCC", linewidth=1.5)

        # Edge legend (right of center)
        legend_right_x = 30
        edge_legend_title = "Pressing Affinity" if use_affinity_weights else "Co-Press Strength"
        ax.text(legend_right_x, legend_y + 5, edge_legend_title, ha="center", fontsize=9, weight="bold", color="#444444")

        strengths = [0.25, 0.6, 1.0]
        edge_labels = ["Weak", "Moderate", "Strong"]
        start_x2 = legend_right_x - 10

        for i, (f, lab) in enumerate(zip(strengths, edge_labels)):
            lw = s.edge_w_min + s.edge_w_max_add * f
            a = s.edge_alpha_min + (s.edge_alpha_max - s.edge_alpha_min) * f
            
            mid = start_x2 + i * 10
            ax.plot([mid - 3, mid + 3], [legend_y, legend_y], linewidth=lw, color=s.edge_color, alpha=a, solid_capstyle="round")
            ax.text(mid, legend_y - 3.5, lab, ha="center", va="top", fontsize=8, color="#666666")

        # Axes and save
        ax.set_ylim(*final_ylim)
        ax.set_xlim(*final_xlim)
        ax.set_aspect("equal", adjustable="box")
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=s.dpi, bbox_inches="tight", facecolor=s.bg_color)
        plt.close()

    def plot_with_centrality(
        self,
        avg_positions: Dict[int, Tuple[float, float]],
        co_press_counts: Dict[Tuple[int, int], int],
        node_sizes: Dict[int, int],
        centrality_metrics: Dict[int, 'CentralityMetrics'],
        player_names: Dict[int, str],
        player_numbers: Dict[int, int],
        player_roles: Dict[int, str],
        highlight_pids: Set[int],
        title: str,
        out_path: str,
        show_all_players: bool = True,
        node_scale: Optional[Tuple[float, float]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot pressing network with centrality visualization.

        Node colors represent betweenness centrality (pressing orchestrators).
        Node borders vary by community membership.

        Args:
            centrality_metrics: Dict[player_id, CentralityMetrics] from PressingNetworkAnalyzer
            Other args same as plot()
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import YlOrRd

        s = self.style

        # Adaptive settings
        backbone_percentile = s.backbone_global_percentile_fullteam if show_all_players else s.backbone_global_percentile_frequent
        top_name_labels = s.top_name_labels_fullteam if show_all_players else s.top_name_labels_frequent

        # Figure
        fig, ax = plt.subplots(figsize=s.figsize)
        fig.set_facecolor(s.bg_color)
        ax.set_facecolor(s.bg_color)
        draw_pitch(ax, color=s.bg_color, line_color=s.pitch_line_color)

        final_xlim = xlim if xlim is not None else s.xlim
        final_ylim = ylim if ylim is not None else s.ylim

        # Nodes included
        valid_nodes = set(avg_positions.keys())
        if not show_all_players:
            valid_nodes = valid_nodes.intersection(set(highlight_pids))

        # Node scaling
        size_vals = np.array([float(node_sizes.get(pid, 0.0)) for pid in valid_nodes], dtype=float)
        if node_scale is None:
            smin, smax = self._robust_norm(size_vals, 5.0, 95.0)
        else:
            smin, smax = node_scale
        size_norm = Normalize(vmin=smin, vmax=smax)

        # Betweenness centrality color mapping
        betweenness_values = [
            centrality_metrics.get(pid, type('obj', (object,), {'betweenness_centrality': 0.0})()).betweenness_centrality
            for pid in valid_nodes
        ]
        if betweenness_values and max(betweenness_values) > 0:
            centrality_norm = Normalize(vmin=min(betweenness_values), vmax=max(betweenness_values))
            cmap = YlOrRd
        else:
            centrality_norm = None
            cmap = None

        # Build graph
        G = nx.Graph()
        for pid in valid_nodes:
            pos = avg_positions[pid]
            size_val = node_sizes.get(pid, 0.0)
            G.add_node(pid, pos=pos, size=size_val, z_ord=3 if pid in highlight_pids else 2)

        # Backbone edge selection
        keep_edges = self._backbone_edges_topk_percentile(
            co_press_counts,
            valid_nodes,
            k_per_node=s.backbone_k_per_node,
            global_percentile=backbone_percentile,
            guarantee_one_edge_per_node=s.guarantee_one_edge_per_node
        )

        for (u, v), w in co_press_counts.items():
            if u not in valid_nodes or v not in valid_nodes:
                continue
            key = (min(u, v), max(u, v))
            if key not in keep_edges:
                continue
            G.add_edge(u, v, weight=float(w))

        # Rank-based edge scaling
        edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0.0))
        n_edges = len(edges_sorted)
        rank_map: Dict[Tuple[int, int], float] = {}

        if n_edges <= 1:
            for (u, v, _) in edges_sorted:
                rank_map[(min(u, v), max(u, v))] = 1.0
        else:
            for i, (u, v, _) in enumerate(edges_sorted):
                rank_map[(min(u, v), max(u, v))] = i / (n_edges - 1)

        # Draw edges
        for (u, v, _) in G.edges(data=True):
            nw = rank_map.get((min(u, v), max(u, v)), 0.0)
            width = s.edge_w_min + s.edge_w_max_add * nw
            alpha = s.edge_alpha_min + (s.edge_alpha_max - s.edge_alpha_min) * nw

            (x1, y1) = self._jitter_pos(G.nodes[u]["pos"], u)
            (x2, y2) = self._jitter_pos(G.nodes[v]["pos"], v)

            ax.plot(
                [x1, x2], [y1, y2],
                linewidth=width,
                color=s.edge_color,
                alpha=alpha,
                zorder=1,
                linestyle="solid",
                solid_capstyle="round"
            )

        # Label policy
        sorted_by_size = sorted(valid_nodes, key=lambda pid: node_sizes.get(pid, 0), reverse=True)
        top_name_ids = set(sorted_by_size[:top_name_labels]).union(set(highlight_pids))

        # Draw nodes with centrality colors
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]["z_ord"])

        for node_id, attrs in sorted_nodes:
            (x, y) = self._jitter_pos(attrs["pos"], node_id)
            val = attrs["size"]
            z_ord = attrs["z_ord"]

            # Calculate radius from size
            frac = np.sqrt(max(val - smin, 0) / (smax - smin + 1e-9))
            r = s.node_r_min + s.node_r_max_add * frac

            # Color by betweenness centrality
            if centrality_norm and cmap and node_id in centrality_metrics:
                betweenness = centrality_metrics[node_id].betweenness_centrality
                color = cmap(centrality_norm(betweenness))
            else:
                color = s.red_constant

            # Border style by community
            if node_id in centrality_metrics:
                comm_id = centrality_metrics[node_id].community_id
                linestyle = ['solid', 'dashed', 'dotted'][comm_id % 3]
            else:
                linestyle = 'solid'

            # Highlight vs non-highlight styling
            if node_id in highlight_pids:
                edge_color = "black"
                edge_width = 1.6
            else:
                edge_color = "#AAAAAA"
                edge_width = 1.0

            ax.add_patch(Circle(
                (x, y),
                radius=r,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                linestyle=linestyle,
                alpha=0.95,
                zorder=z_ord
            ))

            # Number labels (all nodes)
            num = player_numbers.get(node_id, node_id)
            ax.text(
                x, y, str(num),
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                zorder=z_ord + 1
            )

            # Name labels (top players only)
            if node_id in top_name_ids:
                name = player_names.get(node_id, str(node_id))
                txt = ax.text(
                    x, y + r + 1.5,
                    name,
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    color=s.text_color,
                    weight="bold",
                    zorder=z_ord + 1
                )
                txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=s.bg_color)])

        # Title + logo
        logo_x, logo_y = -52.5, 42
        title_x = -48

        logo_path = "data/assets/rm_logo.png"
        try:
            logo = Image.open(logo_path)
            imagebox = OffsetImage(logo, zoom=0.15)
            ab = AnnotationBbox(imagebox, (logo_x, logo_y), frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)
            title_x = logo_x + 12.0
        except Exception:
            title_x = logo_x

        ax.text(title_x, logo_y, title, fontsize=20, weight="bold", color=s.text_color, ha="left", va="center")
        ax.text(title_x, logo_y - 3.5, "Real Madrid Pressing Network | 23/24", fontsize=12, color="#555555", ha="left", va="center")

        # Legend
        legend_y = -46  # Moved down for better spacing
        ax.text(0, legend_y + 8, "CENTRALITY METRICS", ha="center", fontsize=11, weight="bold", color="#222222")

        # Left: Betweenness Centrality (color legend)
        legend_left_x = -30
        ax.text(legend_left_x, legend_y + 5, "Betweenness (Orchestrator)", ha="center", fontsize=9, weight="bold", color="#444444")

        # Show color gradient
        if centrality_norm and cmap:
            gradient_vals = [0.2, 0.6, 1.0]
            gradient_labels = ["Low", "Med", "High"]
            start_x = legend_left_x - 10

            for i, (val, lab) in enumerate(zip(gradient_vals, gradient_labels)):
                cx = start_x + i * 10
                col = cmap(val)
                ax.add_patch(Circle((cx, legend_y), radius=2.0, facecolor=col, edgecolor="black", linewidth=1.2, alpha=0.95))
                ax.text(cx, legend_y - 3.5, lab, ha="center", va="top", fontsize=8, color="#666666")

        # Separator
        ax.plot([0, 0], [legend_y - 6, legend_y + 6], color="#CCCCCC", linewidth=1.5)

        # Right: Pressing Units (border styles)
        legend_right_x = 30
        ax.text(legend_right_x, legend_y + 5, "Pressing Units (Border Style)", ha="center", fontsize=9, weight="bold", color="#444444")

        community_styles = ['solid', 'dashed', 'dotted']
        community_labels = ["First Wave", "Second Wave", "Recovery"]
        start_x2 = legend_right_x - 12

        for i, (style, lab) in enumerate(zip(community_styles, community_labels)):
            cx = start_x2 + i * 10
            ax.add_patch(Circle((cx, legend_y), radius=2.0, facecolor="#D7191C", edgecolor="black",
                               linewidth=1.6, linestyle=style, alpha=0.95))
            ax.text(cx, legend_y - 3.5, lab, ha="center", va="top", fontsize=8, color="#666666")

        # Axes and save
        ax.set_ylim(*final_ylim)
        ax.set_xlim(*final_xlim)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(out_path, dpi=s.dpi, bbox_inches="tight", facecolor=s.bg_color)
        plt.close()
