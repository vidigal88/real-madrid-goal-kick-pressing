"""Centralized time conversion utilities for tracking data analysis.

This module provides utilities for converting between different time formats used in
tracking data analysis:
- Dynamic time format: MM:SS.S (e.g., "12:34.5")
- Tracking time format: HH:MM:SS.DD (e.g., "00:12:34.50")
- Seconds: float (e.g., 754.5)

All time utilities across the codebase should import from this module to ensure
consistency and avoid duplication.

**Common Use Cases:**

1. Convert event timestamps to seconds for arithmetic:
   ```python
   from src.utils.time_utils import time_to_seconds
   kick_seconds = time_to_seconds("00:12:34.50")  # 754.5
   ```

2. Add time offsets:
   ```python
   from src.utils.time_utils import add_seconds_to_time
   new_time = add_seconds_to_time("00:12:34.50", 5.0)  # "00:12:39.50"
   ```

3. Convert dynamic format to tracking format:
   ```python
   from src.utils.time_utils import dynamic_to_tracking_time
   tracking = dynamic_to_tracking_time("12:34.5")  # "00:12:34.50"
   ```

**Format Specifications:**

- **Dynamic**: MM:SS.S (minutes:seconds.deciseconds)
  - Used in event stream data
  - Example: "5:08.2" = 5 minutes, 8.2 seconds

- **Tracking**: HH:MM:SS.DD (hours:minutes:seconds.centiseconds)
  - Used in tracking data timestamps
  - Example: "00:05:08.20" = 5 minutes, 8.2 seconds

- **Seconds**: float (total seconds)
  - Used for arithmetic and comparisons
  - Example: 308.2 = 5 minutes, 8.2 seconds

**See Also:**

- src/extract/services/time_utils.py: Original implementation
- src/features/services/utils.py: Feature-specific time utilities
"""

from __future__ import annotations
from typing import Union


def dynamic_to_tracking_time(time_str: str, period: int = 1) -> str:
    """Convert dynamic time format (MM:SS.S) to tracking time format (HH:MM:SS.DD).

    Dynamic format is used in event stream data, while tracking format is used
    in position tracking data. This function standardizes to tracking format.

    Args:
        time_str: Time string in dynamic format (e.g., "12:34.5").
            Format: MM:SS.S where S can have 0-2 decimal places.
        period: Match period number. Dynamic times are period-relative,
            tracking timestamps are absolute match-clock times.

    Returns:
        Time string in tracking format (e.g., "00:12:34.50").
        Always returns HH:MM:SS.DD with exactly 2 decimal places.

    Examples:
        >>> dynamic_to_tracking_time("12:34.5")
        '00:12:34.50'
        >>> dynamic_to_tracking_time("5:08.2")
        '00:05:08.20'
        >>> dynamic_to_tracking_time("0:03.75")
        '00:00:03.75'
        >>> dynamic_to_tracking_time("45:00")
        '00:45:00.00'
        >>> dynamic_to_tracking_time("5:08.2", period=2)
        '00:50:08.20'

    Notes:
        - Missing decimals are padded with zeros
        - Dynamic times are relative to the current period
        - Centiseconds (2 decimal places) are zero-padded
    """
    minutes_str, rest = time_str.split(":", 1)
    seconds_str, dec_str = (rest.split(".", 1) + ["0"])[:2]
    dec_str = (dec_str + "0")[:2]  # Ensure 2 decimal places
    minute_i = int(minutes_str)
    period_i = max(int(period), 1)
    period_offsets = {
        1: 0,
        2: 45 * 60,
        3: 90 * 60,
        4: 105 * 60,
        5: 120 * 60,
    }
    base_seconds = period_offsets.get(period_i, 45 * 60 * (period_i - 1))

    # Dynamic timestamps are inconsistent across sources: some are relative to
    # the current period, others already use absolute match time.
    if minute_i >= base_seconds // 60:
        total_seconds = minute_i * 60 + int(seconds_str) + int(dec_str) / 100.0
    else:
        total_seconds = base_seconds + minute_i * 60 + int(seconds_str) + int(dec_str) / 100.0
    return seconds_to_time(total_seconds)


def time_to_seconds(tracking_time: Union[str, float]) -> float:
    """Convert tracking time format (HH:MM:SS.DD) to total seconds.

    This is the primary conversion function for time arithmetic. Use this to
    compute time differences, offsets, and comparisons.

    Args:
        tracking_time: Time string in tracking format (e.g., "00:12:34.50")
            or already a float (passes through).
            Format: HH:MM:SS.DD where DD is centiseconds (0-99).

    Returns:
        Total seconds as float.

    Examples:
        >>> time_to_seconds("00:12:34.50")
        754.5
        >>> time_to_seconds("01:00:00.00")
        3600.0
        >>> time_to_seconds("00:00:05.25")
        5.25
        >>> time_to_seconds(123.45)  # Pass-through
        123.45

    Raises:
        ValueError: If time string format is invalid.

    Notes:
        - If input is already a float, returns it unchanged (idempotent)
        - Centiseconds (DD) are interpreted as hundredths of a second
        - Supports times up to 99:59:59.99 (359999.99 seconds)
    """
    # Pass-through if already a number
    if isinstance(tracking_time, (int, float)):
        return float(tracking_time)

    # Parse tracking format
    try:
        hh, mm, ss_dec = tracking_time.split(":")
        ss, dec = (ss_dec.split(".", 1) + ["0"])[:2]
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(dec) / 100.0
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid tracking time format: {tracking_time}. Expected HH:MM:SS.DD") from e


def seconds_to_time(seconds: float) -> str:
    """Convert total seconds to tracking time format (HH:MM:SS.DD).

    Inverse operation of time_to_seconds(). Use this to format computed times
    back to tracking format for display or storage.

    Args:
        seconds: Total seconds. Negative values are clamped to 0.0.

    Returns:
        Time string in tracking format (HH:MM:SS.DD).

    Examples:
        >>> seconds_to_time(754.5)
        '00:12:34.50'
        >>> seconds_to_time(3661.25)
        '01:01:01.25'
        >>> seconds_to_time(0.5)
        '00:00:00.50'
        >>> seconds_to_time(-10.0)  # Negative clamped to 0
        '00:00:00.00'

    Notes:
        - Negative values are clamped to 0 (no negative times)
        - Rounds to nearest centisecond (0.01s precision)
        - Supports times up to 99:59:59.99
    """
    if seconds < 0:
        seconds = 0.0

    # Convert to centiseconds for integer arithmetic
    total_cs = int(round(seconds * 100))

    hh = total_cs // (3600 * 100)
    total_cs -= hh * 3600 * 100

    mm = total_cs // (60 * 100)
    total_cs -= mm * 60 * 100

    ss = total_cs // 100
    cs = total_cs - ss * 100

    return f"{hh:02d}:{mm:02d}:{ss:02d}.{cs:02d}"


def add_seconds_to_time(tracking_time: str, delta_seconds: float) -> str:
    """Add or subtract seconds from a tracking time.

    Convenience function for time arithmetic. Equivalent to:
    seconds_to_time(time_to_seconds(tracking_time) + delta_seconds)

    Args:
        tracking_time: Time string in tracking format (HH:MM:SS.DD).
        delta_seconds: Seconds to add (positive) or subtract (negative).

    Returns:
        New time string in tracking format.

    Examples:
        >>> add_seconds_to_time("00:12:34.50", 10.0)
        '00:12:44.50'
        >>> add_seconds_to_time("00:12:34.50", -5.0)
        '00:12:29.50'
        >>> add_seconds_to_time("00:00:05.00", -10.0)  # Would be negative, clamped to 0
        '00:00:00.00'
        >>> add_seconds_to_time("00:59:55.00", 10.0)  # Rolls over to next minute
        '01:00:05.00'

    Notes:
        - Negative results are clamped to 00:00:00.00
        - Automatically handles minute/hour rollovers
    """
    return seconds_to_time(time_to_seconds(tracking_time) + float(delta_seconds))


def time_difference(time1: str, time2: str) -> float:
    """Compute the difference between two tracking times in seconds.

    Args:
        time1: First time in tracking format (HH:MM:SS.DD).
        time2: Second time in tracking format (HH:MM:SS.DD).

    Returns:
        Difference in seconds (time1 - time2).
        Positive if time1 > time2, negative if time1 < time2.

    Examples:
        >>> time_difference("00:12:34.50", "00:12:30.00")
        4.5
        >>> time_difference("00:12:30.00", "00:12:34.50")
        -4.5
        >>> time_difference("01:00:00.00", "00:00:00.00")
        3600.0

    Notes:
        - Can return negative values (unlike seconds_to_time which clamps)
        - Use abs() if you need absolute difference
    """
    return time_to_seconds(time1) - time_to_seconds(time2)


# Backward compatibility aliases
def prepare_time_for_arithmetic(time_str: str) -> float:
    """Deprecated: Use time_to_seconds() instead.

    Kept for backward compatibility with legacy code.
    """
    return time_to_seconds(time_str)
