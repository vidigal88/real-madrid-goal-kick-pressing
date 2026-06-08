"""Time conversion utilities for tracking data analysis.

This module provides utilities for converting between dynamic time format (MM:SS.S)
and tracking time format (HH:MM:SS.DD), as well as performing time arithmetic.
"""

from __future__ import annotations


def dynamic_to_tracking_time(time_str: str, period: int = 1) -> str:
    """Convert dynamic time format (MM:SS.S) to tracking time format (HH:MM:SS.DD).

    Args:
        time_str: Time string in dynamic format (e.g., "12:34.5")
        period: Match period number. Dynamic times are relative to the period,
            while tracking timestamps are absolute match-clock times.

    Returns:
        Time string in tracking format (e.g., "00:12:34.50")

    Examples:
        >>> dynamic_to_tracking_time("12:34.5")
        "00:12:34.50"
        >>> dynamic_to_tracking_time("5:08.2")
        "00:05:08.20"
        >>> dynamic_to_tracking_time("5:08.2", period=2)
        "00:50:08.20"
    """
    minutes_str, rest = time_str.split(":", 1)
    seconds_str, dec_str = (rest.split(".", 1) + ["0"])[:2]
    dec_str = (dec_str + "0")[:2]
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

    # Some dynamic feeds use period-relative times (e.g. 18:30 in P2),
    # while others already use the absolute match clock (e.g. 63:00 in P2).
    if minute_i >= base_seconds // 60:
        total_seconds = minute_i * 60 + int(seconds_str) + int(dec_str) / 100.0
    else:
        total_seconds = base_seconds + minute_i * 60 + int(seconds_str) + int(dec_str) / 100.0
    return seconds_to_time(total_seconds)


def time_to_seconds(tracking_time: str) -> float:
    """Convert tracking time format (HH:MM:SS.DD) to total seconds.

    Args:
        tracking_time: Time string in tracking format (e.g., "00:12:34.50")

    Returns:
        Total seconds as float

    Examples:
        >>> time_to_seconds("00:12:34.50")
        754.5
        >>> time_to_seconds("01:00:00.00")
        3600.0
    """
    hh, mm, ss_dec = tracking_time.split(":")
    ss, dec = (ss_dec.split(".", 1) + ["0"])[:2]
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(dec) / 100.0


def seconds_to_time(seconds: float) -> str:
    """Convert total seconds to tracking time format (HH:MM:SS.DD).

    Args:
        seconds: Total seconds (negative values are clamped to 0)

    Returns:
        Time string in tracking format

    Examples:
        >>> seconds_to_time(754.5)
        "00:12:34.50"
        >>> seconds_to_time(3661.25)
        "01:01:01.25"
    """
    if seconds < 0:
        seconds = 0.0
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

    Args:
        tracking_time: Time string in tracking format
        delta_seconds: Seconds to add (negative to subtract)

    Returns:
        New time string in tracking format

    Examples:
        >>> add_seconds_to_time("00:12:34.50", 10.0)
        "00:12:44.50"
        >>> add_seconds_to_time("00:12:34.50", -5.0)
        "00:12:29.50"
    """
    return seconds_to_time(time_to_seconds(tracking_time) + float(delta_seconds))
