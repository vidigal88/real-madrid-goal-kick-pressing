"""Data extraction module for Real Madrid pressing analysis.

This module provides functionality to extract and process opponent goal-kick build-up
moments from SkillCorner tracking data.
"""

from .extraction import run_extractor_to_disk

__all__ = ["run_extractor_to_disk"]
