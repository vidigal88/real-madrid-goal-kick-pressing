"""Hierarchical clustering of pressing patterns based on NMF topic loadings.

This module groups similar pressing build-ups into clusters by applying agglomerative
clustering to NMF topic weights. Build-ups with similar topic distributions are grouped
together, revealing distinct pressing strategy archetypes.

**Clustering Approach:**

After NMF topic modeling, each build-up is represented as a K-dimensional vector of topic
weights (from H matrix). Build-ups are then clustered based on similarity in topic space,
identifying groups that employ similar combinations of pressing patterns.

**Why Cluster on Topics (not raw tokens)?**

- **Dimensionality**: Topics (K=15) vs tokens (120) → more stable clustering
- **Interpretability**: Clusters represent "meta-strategies" (topic combinations)
- **Noise reduction**: Topics smooth out token-level variability

**Algorithm:**

Uses hierarchical agglomerative clustering with Ward linkage:
1. Start with each build-up as its own cluster
2. Iteratively merge closest clusters based on Ward criterion (minimize variance)
3. Stop when reaching desired number of clusters

**Evaluation:**

Silhouette score measures cluster quality:
- Score range: [-1, 1]
- >0.5: Strong clustering
- 0.3-0.5: Reasonable clustering
- <0.3: Weak clustering (overlapping groups)

**Usage:**

    $ python src/models/clustering.py \\
        --topics-dir data/processed/rm_pressing_topics \\
        --n-clusters 5

    # Output: clusters.parquet with build_up_id → cluster_label mapping

**Interpreting Clusters:**

Clusters group build-ups with similar pressing approaches:
- **Cluster 0**: High-intensity, coordinated press (high Topic 1, Topic 4)
- **Cluster 1**: Reactive, containment-focused (high Topic 7, low intensity)
- **Cluster 2**: Asymmetric left-side press (high Topic 3)
- etc.

**See Also:**

- src/models/nmf_topics.py: Generates input H matrix
- src/analysis/narratives.py: Creates cluster descriptions
- docs/concepts/clustering.md: Clustering methodology guide
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from src.models.config import ClusteringConfig
from typing import Dict, Any

def cluster_build_ups(h_matrix: pd.DataFrame, config: ClusteringConfig) -> Dict[str, Any]:
    """Cluster build-ups based on NMF topic weights using hierarchical clustering.

    Groups build-ups with similar pressing patterns by clustering in topic space.
    Uses agglomerative clustering with Ward linkage for hierarchical grouping.

    Args:
        h_matrix: NMF H matrix with shape (K topics × N build_ups).
            Columns: build_up_id
            Rows: Topic indices (0 to K-1)
            Values: Topic weights (higher = more prevalent in that build-up)

        config: ClusteringConfig containing:
            - n_clusters: Target number of clusters (default: 5)
            - linkage: Linkage criterion ('ward', 'complete', 'average')

    Returns:
        Dictionary containing:
        - 'labels': NumPy array of cluster assignments (length N)
        - 'silhouette_score': Cluster quality metric (float, range [-1, 1])
        - 'n_clusters': Number of clusters used

    Example:
        >>> H = pd.read_parquet("H.parquet")  # (15 topics × 94 build_ups)
        >>> config = ClusteringConfig(n_clusters=5, linkage='ward')
        >>> result = cluster_build_ups(H, config)
        >>> print(f"Silhouette score: {result['silhouette_score']:.3f}")
        >>> print(f"Cluster distribution: {np.bincount(result['labels'])}")
    """
    # H is (Topics K x Build-ups N)
    X = h_matrix.T # (N x K)
    
    clustering = AgglomerativeClustering(
        n_clusters=config.n_clusters,
        linkage=config.linkage
    )
    
    labels = clustering.fit_predict(X)
    
    # Silhouette
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
    else:
        score = 0.0
        
    return {
        "labels": labels,
        "silhouette_score": float(score),
        "n_clusters": config.n_clusters
    }

def main() -> None:
    """Main entry point for clustering build-ups from command line.

    Loads NMF H matrix, performs hierarchical clustering, and exports results
    to parquet file with cluster assignments and quality metrics.
    """
    import argparse
    from pathlib import Path
    from src.models.config import ClusteringConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-dir", default="data/processed/rm_pressing_topics")
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()
    
    topics_dir = Path(args.topics_dir)
    h_path = topics_dir / "H.parquet"
    
    if not h_path.exists():
        print("H matrix not found. Run nmf_topics.py first.")
        return
        
    H = pd.read_parquet(h_path)
    config = ClusteringConfig(n_clusters=args.n_clusters)
    
    res = cluster_build_ups(H, config)
    
    # Save clusters
    labels_df = pd.DataFrame({"build_up_id": H.columns, "cluster_id": res["labels"]})
    labels_df.to_parquet(topics_dir / "clusters.parquet", index=False)
    
    print(f"Clustering complete. Silhouette Score: {res['silhouette_score']:.3f}")

if __name__ == "__main__":
    main()

