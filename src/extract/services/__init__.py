"""Service modules for extraction pipeline.

This package contains specialized service modules for different aspects of the
extraction pipeline:

- build_up_detector: Core build-up phase detection logic
- data_loader: Loading and converting tracking data
- geometry: Geometric calculations for pitch analysis
- goal_kick_detector: Goal kick event detection
- team_utils: Team and player identification utilities
- time_utils: Time conversion and manipulation
"""

from .build_up_detector import BuildUp, detect_build_up_from_reference
from .data_loader import (
    list_full_game_ids,
    load_dynamic,
    load_meta,
    load_tracking_auto,
    load_tracking_parquet,
)
from .geometry import ball_in_goal_area, distance, goal_area_x_bounds
from .goal_kick_detector import (
    GoalKickRef,
    detect_goalkeeper_side_from_parquet,
    filter_goal_kick_refs,
)
from .team_utils import find_real_madrid_and_opponent, find_starting_goalkeeper
from .time_utils import (
    add_seconds_to_time,
    dynamic_to_tracking_time,
    seconds_to_time,
    time_to_seconds,
)

__all__ = [
    # build_up_detector
    "BuildUp",
    "detect_build_up_from_reference",
    # data_loader
    "list_full_game_ids",
    "load_dynamic",
    "load_meta",
    "load_tracking_auto",
    "load_tracking_parquet",
    # geometry
    "ball_in_goal_area",
    "distance",
    "goal_area_x_bounds",
    # goal_kick_detector
    "GoalKickRef",
    "detect_goalkeeper_side_from_parquet",
    "filter_goal_kick_refs",
    # team_utils
    "find_real_madrid_and_opponent",
    "find_starting_goalkeeper",
    # time_utils
    "add_seconds_to_time",
    "dynamic_to_tracking_time",
    "seconds_to_time",
    "time_to_seconds",
]
