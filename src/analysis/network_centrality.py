"""Network centrality analysis for pressing partnership graphs.

This module applies graph theory metrics to pressing networks to quantify player roles
and network structure. Centrality measures reveal:

**Player-Level Metrics:**

1. **Degree Centrality**: Number of pressing partnerships
   - High degree → well-connected presser, presses with many teammates
   - Low degree → specialist presser, limited partnerships
   - Range: [0, 1], normalized by (N-1)

2. **Betweenness Centrality**: Player's importance as bridge between pressing units
   - High betweenness → pressing orchestrator, connects subgroups
   - Low betweenness → peripheral presser
   - Identifies "connector" players linking defensive and midfield lines

3. **Clustering Coefficient**: How tightly connected a player's partners are
   - High clustering → part of tight pressing unit (partners also press together)
   - Low clustering → bridges disparate units
   - Range: [0, 1]

4. **Community ID**: Detected pressing subgroup (Louvain algorithm)
   - Players in same community press together frequently
   - Reveals pressing units (e.g., left-side trio, central block)

**Network-Level Metrics:**

1. **Density**: Overall connectivity (% of possible edges present)
   - High density → cohesive team-wide pressing
   - Low density → fragmented pressing structure

2. **Average Clustering**: Team-wide pressing coordination
   - High → many tight subgroups (triangle patterns)
   - Low → linear pressing chains

3. **Number of Communities**: Pressing structure granularity
   - Few communities → centralized pressing
   - Many communities → specialized pressing units

4. **Average Path Length**: Typical separation between players
   - Short paths → well-integrated network
   - Long paths → hierarchical structure

**Usage:**

    from src.analysis.network_centrality import PressingNetworkAnalyzer

    analyzer = PressingNetworkAnalyzer(
        avg_positions=player_positions,
        co_press_counts=partnership_counts,
        node_sizes=press_frequencies
    )

    # Analyze network
    player_metrics = analyzer.calculate_centrality()
    network_stats = analyzer.calculate_network_metrics()

    # Identify key players
    orchestrators = sorted(
        player_metrics,
        key=lambda x: x.betweenness_centrality,
        reverse=True
    )[:3]

**Applications:**

- **Role Identification**: Classify players as orchestrators, connectors, specialists
- **Tactical Insight**: Understand pressing structure (centralized vs modular)
- **Performance Tracking**: Monitor centrality changes across matches
- **Strategic Planning**: Target key nodes for opponent disruption

**See Also:**

- src/models/pressing_affinity.py: Computes edge weights (affinity)
- src/viz/network.py: Visualizes networks with centrality-based styling
- docs/concepts/network-analysis.md: Graph theory methodology
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class CentralityMetrics:
    """Centrality metrics for a single player."""
    player_id: int
    degree_centrality: float
    betweenness_centrality: float
    clustering_coefficient: float
    community_id: int


@dataclass
class NetworkMetrics:
    """Overall network statistics."""
    density: float  # How connected the network is
    avg_clustering: float  # Average clustering coefficient
    num_communities: int  # Number of detected communities
    avg_path_length: float  # Typical separation between players


class PressingNetworkAnalyzer:
    """Analyzes pressing network structure using graph theory."""

    def __init__(
        self,
        avg_positions: Dict[int, Tuple[float, float]],
        co_press_counts: Dict[Tuple[int, int], int],
        node_sizes: Dict[int, int]
    ):
        """
        Initialize the analyzer with pressing network data.

        Args:
            avg_positions: {pid: (x, y)} average pressing positions
            co_press_counts: {(pid1, pid2): count} co-pressing frequencies
            node_sizes: {pid: count} individual press counts
        """
        self.positions = avg_positions
        self.edges = co_press_counts
        self.node_sizes = node_sizes
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """Constructs NetworkX graph from pressing data."""
        G = nx.Graph()

        # Add nodes with attributes
        for pid, (x, y) in self.positions.items():
            G.add_node(
                pid,
                pos=(x, y),
                size=self.node_sizes.get(pid, 0)
            )

        # Add edges with weights
        for (u, v), weight in self.edges.items():
            if u in G.nodes and v in G.nodes:
                G.add_edge(u, v, weight=weight)

        return G

    def calculate_centrality(self) -> Dict[int, CentralityMetrics]:
        """
        Calculates all centrality metrics per player.

        Returns:
            Dictionary mapping player_id to CentralityMetrics

        Example:
            >>> analyzer = PressingNetworkAnalyzer(positions, edges, sizes)
            >>> metrics = analyzer.calculate_centrality()
            >>> tchouameni_metrics = metrics[123]  # player_id 123
            >>> print(f"Betweenness: {tchouameni_metrics.betweenness_centrality:.3f}")
        """
        if len(self.graph.nodes) == 0:
            return {}

        # Degree centrality (normalized by max possible connections)
        degree = nx.degree_centrality(self.graph)

        # Betweenness centrality (uses shortest paths)
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')

        # Clustering coefficient (local transitivity)
        clustering = nx.clustering(self.graph)

        # Community detection (Louvain algorithm)
        communities = nx.community.louvain_communities(self.graph, weight='weight')
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i

        # Package results
        metrics = {}
        for pid in self.graph.nodes:
            metrics[pid] = CentralityMetrics(
                player_id=pid,
                degree_centrality=degree.get(pid, 0.0),
                betweenness_centrality=betweenness.get(pid, 0.0),
                clustering_coefficient=clustering.get(pid, 0.0),
                community_id=community_map.get(pid, -1)
            )

        return metrics

    def calculate_network_metrics(self) -> NetworkMetrics:
        """
        Calculates overall network statistics.

        Returns:
            NetworkMetrics with density, clustering, communities, avg path length
        """
        if len(self.graph.nodes) == 0:
            return NetworkMetrics(0.0, 0.0, 0, 0.0)

        # Density: actual edges / possible edges
        density = nx.density(self.graph)

        # Average clustering
        avg_clustering = nx.average_clustering(self.graph)

        # Communities
        communities = nx.community.louvain_communities(self.graph, weight='weight')
        num_communities = len(communities)

        # Average path length (only if connected)
        if nx.is_connected(self.graph):
            avg_path = nx.average_shortest_path_length(self.graph, weight='weight')
        else:
            # Use largest component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            if len(largest_cc) > 1:
                subgraph = self.graph.subgraph(largest_cc)
                avg_path = nx.average_shortest_path_length(subgraph, weight='weight')
            else:
                avg_path = 0.0

        return NetworkMetrics(
            density=density,
            avg_clustering=avg_clustering,
            num_communities=num_communities,
            avg_path_length=avg_path
        )

    def identify_key_players(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Returns top players by different metrics.

        Returns:
            Dictionary with keys:
            - 'orchestrators': Top 3 by betweenness (pressing coordinators)
            - 'hubs': Top 3 by degree (most connected pressers)
            - 'tight_units': Top 3 by clustering (players in cohesive units)

        Example:
            >>> key_players = analyzer.identify_key_players()
            >>> print("Top pressing orchestrators:")
            >>> for pid, score in key_players['orchestrators']:
            ...     print(f"  Player {pid}: {score:.3f}")
        """
        centrality = self.calculate_centrality()

        if not centrality:
            return {
                'orchestrators': [],
                'hubs': [],
                'tight_units': []
            }

        # Sort by each metric
        by_betweenness = sorted(
            [(m.player_id, m.betweenness_centrality) for m in centrality.values()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        by_degree = sorted(
            [(m.player_id, m.degree_centrality) for m in centrality.values()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        by_clustering = sorted(
            [(m.player_id, m.clustering_coefficient) for m in centrality.values()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            'orchestrators': by_betweenness,
            'hubs': by_degree,
            'tight_units': by_clustering
        }

    def export_centrality_report(
        self,
        player_names: Dict[int, str],
        output_path: str
    ) -> None:
        """
        Exports detailed centrality analysis to CSV.

        Args:
            player_names: {pid: name} mapping for readable output
            output_path: Path to save CSV file

        Output Format:
            CSV with columns: player_id, player_name, press_count, degree_centrality,
            betweenness_centrality, clustering_coefficient, community_id

            Network-level stats appended as comment footer

        Example:
            >>> analyzer.export_centrality_report(
            ...     player_names={123: "Tchouaméni", 456: "Kroos"},
            ...     output_path="centrality_metrics.csv"
            ... )
        """
        centrality = self.calculate_centrality()
        network_metrics = self.calculate_network_metrics()

        # Player-level metrics
        rows = []
        for pid, metrics in centrality.items():
            rows.append({
                'player_id': pid,
                'player_name': player_names.get(pid, str(pid)),
                'press_count': self.node_sizes.get(pid, 0),
                'degree_centrality': metrics.degree_centrality,
                'betweenness_centrality': metrics.betweenness_centrality,
                'clustering_coefficient': metrics.clustering_coefficient,
                'community_id': metrics.community_id
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('betweenness_centrality', ascending=False)
        df.to_csv(output_path, index=False)

        # Network-level metrics (append as comment footer)
        with open(output_path, 'a') as f:
            f.write(f"\n# Network Statistics\n")
            f.write(f"# Density: {network_metrics.density:.3f}\n")
            f.write(f"# Avg Clustering: {network_metrics.avg_clustering:.3f}\n")
            f.write(f"# Communities: {network_metrics.num_communities}\n")
            f.write(f"# Avg Path Length: {network_metrics.avg_path_length:.2f}\n")

    def get_community_members(self) -> Dict[int, List[int]]:
        """
        Returns members of each detected community.

        Returns:
            {community_id: [player_ids]} mapping

        Example:
            >>> communities = analyzer.get_community_members()
            >>> print(f"Left-side pressing unit: {communities[0]}")
            >>> print(f"Right-side pressing unit: {communities[1]}")
        """
        centrality = self.calculate_centrality()

        comm_members = {}
        for pid, metrics in centrality.items():
            comm_id = metrics.community_id
            if comm_id not in comm_members:
                comm_members[comm_id] = []
            comm_members[comm_id].append(pid)

        return comm_members
