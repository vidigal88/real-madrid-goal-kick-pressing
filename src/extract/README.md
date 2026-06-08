# Extract Module

This module provides functionality to extract and process opponent goal-kick build-up moments from SkillCorner tracking data for Real Madrid pressing analysis.

## Structure

```
src/extract/
├── extraction.py           # Main orchestrator script
├── services/               # Service modules
│   ├── __init__.py        # Package exports
│   ├── build_up_detector.py    # Build-up phase detection logic
│   ├── data_loader.py          # Data loading utilities
│   ├── geometry.py             # Geometric calculations
│   ├── goal_kick_detector.py   # Goal kick detection
│   ├── team_utils.py           # Team/player identification
│   └── time_utils.py           # Time conversion utilities
└── README.md              # This file
```

## Service Modules

### build_up_detector.py
Core logic for detecting build-up phases from tracking data:
- `BuildUp`: Data class representing a complete build-up phase
- `detect_build_up_from_reference()`: Main detection function
- `merge_ball_and_gk()`: Merge ball and goalkeeper positions
- `find_setup_segments()`: Find continuous setup segments
- `detect_kick_time_from_setup()`: Detect kick moment
- `find_ready_time_in_setup()`: Identify goalkeeper ready state

### data_loader.py
Handles loading and converting various data formats:
- `load_meta()`: Load match metadata JSON
- `load_dynamic()`: Load event data parquet
- `load_tracking_parquet()`: Load tracking parquet
- `load_tracking_auto()`: Automatically load tracking (parquet or JSON)
- `convert_tracking_json_to_long_df()`: Convert JSON to long-format DataFrame
- `list_full_game_ids()`: List games with complete data

### geometry.py
Geometric calculations for pitch analysis:
- `distance()`: Euclidean distance between points
- `goal_area_x_bounds()`: Calculate goal area X-axis bounds
- `ball_in_goal_area()`: Check if ball is in goal area

### goal_kick_detector.py
Goal kick event detection and analysis:
- `GoalKickRef`: Data class for goal kick references
- `filter_goal_kick_refs()`: Extract goal kick events from dynamic data
- `detect_goalkeeper_side_from_parquet()`: Determine GK defending side

### team_utils.py
Team and player identification:
- `find_real_madrid_and_opponent()`: Identify Real Madrid and opponent
- `find_starting_goalkeeper()`: Find starting goalkeeper for a team

### time_utils.py
Time conversion and manipulation:
- `dynamic_to_tracking_time()`: Convert MM:SS.S to HH:MM:SS.DD
- `time_to_seconds()`: Convert tracking time to seconds
- `seconds_to_time()`: Convert seconds to tracking time
- `add_seconds_to_time()`: Add/subtract seconds from time

## Usage

### Command Line

```bash
# Process default 5 games
python src/extract/extraction.py

# Process all available games
python src/extract/extraction.py --full

# Process specific games
python src/extract/extraction.py 2014987 2016604

# Custom parameters
python src/extract/extraction.py --lookback-seconds 120 --window-after-seconds 15 --verbose
```

### As Module

```python
from src.extract import run_extractor_to_disk
from pathlib import Path

result = run_extractor_to_disk(
    data_root=Path("data/raw/RealMadrid"),
    out_dir=Path("data/processed/rm_pressing"),
    game_ids=["2014987", "2016604"],
    lookback_seconds=90,
    window_after_seconds=10,
    pre_kick_seconds=2.0,
    start_from_ready=True,
    max_seconds_before_ref=15.0,
    goal_area_depth_m=5.5,
    goal_area_half_width_m=10.0,
    goal_area_x_margin_m=1.0,
    gk_ball_distance_m=6.0,
    kick_displacement_m=1.0,
    kick_confirm_frames=5,
    cache_tracking_parquet=False,
    rebuild_tracking_parquet=False,
    verbose=True,
)
print(f"Extracted {result['n_build_ups']} build-ups")
```

## Inputs

The extraction process expects data under `--data-root` (default: `data/raw/RealMadrid`):
- `meta/<match_id>.json` - Match metadata
- `dynamic/<match_id>.parquet` - Event data
- `tracking_parquet/<match_id>.parquet` OR `tracking/<match_id>.json` - Tracking data

## Outputs

Results are saved to `--out-dir` (default: `data/processed/rm_pressing`):
- `index.parquet` - One row per extracted build-up with metadata
- `frames/build_up_XXXXXXX.parquet` - Tracking data for each build-up window
- `params.json` - Extraction parameters used

## Detection Parameters

### Time Windows
- `lookback_seconds`: How far back to search before goal kick event (default: 90)
- `window_after_seconds`: Seconds to include after kick/ready (default: 10)
- `pre_kick_seconds`: Seconds before kick if not start_from_ready (default: 2.0)
- `max_seconds_before_ref`: Maximum seconds before event to accept (default: 15.0)

### Geometry
- `goal_area_depth_m`: Goal area depth from goal line (default: 5.5)
- `goal_area_half_width_m`: Goal area half-width (default: 10.0)
- `goal_area_x_margin_m`: Extra margin for goal area (default: 1.0)
- `gk_ball_distance_m`: Max GK-ball distance for setup (default: 6.0)

### Kick Detection
- `kick_displacement_m`: Min displacement to detect kick (default: 1.0)
- `kick_confirm_frames`: Frames to confirm kick (default: 5)

### Ready State (hardcoded in build_up_detector)
- `ready_min_dist_m`: Min GK-ball distance for ready (default: 2.0)
- `ready_max_dist_m`: Max GK-ball distance for ready (default: 5.0)
- `ready_stable_frames`: Frames needed for stable ready state (default: 3)
- `ready_ball_step_eps_m`: Max ball movement for ready (default: 0.5)
- `ready_gk_step_eps_m`: Max GK movement for ready (default: 1.0)

## Development

All service modules are well-documented with:
- Detailed docstrings for all functions
- Type hints for parameters and return values
- Usage examples in docstrings
- Clear separation of concerns

To extend the module:
1. Add new functions to appropriate service modules
2. Update service `__init__.py` if adding public APIs
3. Document new functionality in docstrings
4. Follow existing code style and patterns
