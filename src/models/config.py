from dataclasses import dataclass, field
from typing import List

@dataclass
class GMMConfig:
    n_initial_zones: int = 8
    n_target_zones: int = 15
    presser_top_k: int = 5
    presser_min_frames: int = 10
    covariance_type: str = "full"
    random_state: int = 42
    
@dataclass
class NMFConfig:
    n_topics: int = 15
    init: str = "nndsvd"
    alpha: float = 0.0
    l1_ratio: float = 0.0
    random_state: int = 42

@dataclass
class ClusteringConfig:
    n_clusters: int = 5
    linkage: str = "ward"

@dataclass
class ModelConfig:
    gmm: GMMConfig = field(default_factory=GMMConfig)
    nmf: NMFConfig = field(default_factory=NMFConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
