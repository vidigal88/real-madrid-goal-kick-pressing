from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class NormalizationConfig:
    pitch_length: float = 105.0
    pitch_width: float = 68.0

@dataclass
class PossessionConfig:
    possession_radius_m: float = 8.0  # Increased from 1.5m to handle goal-kick scenarios where players are spread out
    temporal_smoothing_window: int = 5

@dataclass
class GoalKickConfig:
    min_stable_frames: int = 3
    stable_movement_threshold_m: float = 0.5
    short_restart_threshold_m: float = 15.0
    long_kick_threshold_m: float = 30.0
    lane_width_m: float = 20.0  # Center lane is -10 to 10

@dataclass
class PressureConfig:
    pressure_distance_m: float = 3.0
    pressure_extended_distance_m: float = 5.0
    pressure_closing_speed_mps: float = 1.0
    burst_gap_frames: int = 2

@dataclass
class CompactnessConfig:
    pass

@dataclass
class SteeringConfig:
    wide_channel_threshold_m: float = 20.0

@dataclass
class OutcomeConfig:
    long_ball_threshold_m: float = 30.0
    touchline_threshold_m: float = 32.0  # Near out of play
    outcome_window_s: float = 2.0

@dataclass
class QCConfig:
    min_ball_detected_ratio: float = 0.8
    min_avg_players: float = 18.0
    max_carrier_missing_ratio: float = 0.3

@dataclass
class FeatureConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    goal_kick: GoalKickConfig = field(default_factory=GoalKickConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
    compactness: CompactnessConfig = field(default_factory=CompactnessConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    qc: QCConfig = field(default_factory=QCConfig)
