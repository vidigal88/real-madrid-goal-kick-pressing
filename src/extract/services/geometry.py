"""Geometric calculations for football pitch analysis.

This module provides utilities for calculating distances, determining goal area bounds,
and checking if positions are within specific pitch regions.
"""

from __future__ import annotations

import math


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        ax: X coordinate of point A
        ay: Y coordinate of point A
        bx: X coordinate of point B
        by: Y coordinate of point B

    Returns:
        Euclidean distance between the two points

    Examples:
        >>> distance(0, 0, 3, 4)
        5.0
        >>> distance(1, 1, 1, 1)
        0.0
    """
    return math.hypot(ax - bx, ay - by)


def goal_area_x_bounds(
    half_length_m: float,
    goal_area_depth_m: float,
    margin_m: float,
    side: str
) -> tuple[float, float]:
    """Calculate the X-axis bounds of the goal area for a given side.

    Args:
        half_length_m: Half the pitch length (e.g., 52.5m for 105m pitch)
        goal_area_depth_m: Depth of the goal area from the goal line
        margin_m: Additional margin to extend the goal area
        side: Either "left" or "right" indicating which goal

    Returns:
        Tuple of (x_min, x_max) defining the goal area bounds

    Examples:
        >>> goal_area_x_bounds(52.5, 5.5, 1.0, "right")
        (45.0, 52.5)
        >>> goal_area_x_bounds(52.5, 5.5, 1.0, "left")
        (-52.5, -45.0)
    """
    if side == "right":
        return half_length_m - (goal_area_depth_m + margin_m), half_length_m
    return -half_length_m, -half_length_m + (goal_area_depth_m + margin_m)


def ball_in_goal_area(
    ball_x: float,
    ball_y: float,
    x_min: float,
    x_max: float,
    half_width_m: float
) -> bool:
    """Check if the ball is within the goal area bounds.

    Args:
        ball_x: Ball X coordinate
        ball_y: Ball Y coordinate
        x_min: Minimum X bound of goal area
        x_max: Maximum X bound of goal area
        half_width_m: Half the width of the goal area (Y-axis)

    Returns:
        True if ball is within the goal area, False otherwise

    Examples:
        >>> ball_in_goal_area(50.0, 5.0, 45.0, 52.5, 10.0)
        True
        >>> ball_in_goal_area(40.0, 5.0, 45.0, 52.5, 10.0)
        False
        >>> ball_in_goal_area(50.0, 15.0, 45.0, 52.5, 10.0)
        False
    """
    return x_min <= ball_x <= x_max and abs(ball_y) <= half_width_m
