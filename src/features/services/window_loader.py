"""Extracted build-up data loading utilities.

This module provides the WindowLoader class for loading extracted build-up
windows from preprocessed parquet files, along with their metadata index.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class WindowLoader:
    """Loader for extracted build-up tracking windows.

    Manages access to extracted build-up frames and metadata stored in
    parquet format. Provides lazy loading of the index and efficient
    retrieval of individual build-up windows.

    Attributes:
        processed_root: Root directory containing extracted data.
        frames_dir: Directory containing individual build-up frame files.
        index_path: Path to the index parquet file with metadata.

    Example:
        >>> loader = WindowLoader("data/processed/rm_pressing")
        >>> df = loader.load_build_up(42)
        >>> meta = loader.get_metadata(42)
        >>> print(meta['kick_time'])
        '00:05:23.50'
    """

    def __init__(self, processed_root: str) -> None:
        """Initialize the WindowLoader.

        Args:
            processed_root: Root directory path containing 'frames/' subdirectory
                and 'index.parquet' file.

        Raises:
            FileNotFoundError: When index file is accessed but doesn't exist.
        """
        self.processed_root = Path(processed_root)
        self.frames_dir = self.processed_root / "frames"
        self.index_path = self.processed_root / "index.parquet"
        self._index: Optional[pd.DataFrame] = None

    @property
    def index(self) -> pd.DataFrame:
        """Lazy-loaded index DataFrame with build-up metadata.

        Returns:
            DataFrame with columns: build_up_id, game_id, kick_time, ready_time,
                outcome, and other metadata fields.

        Raises:
            FileNotFoundError: If index file doesn't exist at index_path.

        Example:
            >>> loader = WindowLoader("data/processed/rm_pressing")
            >>> print(loader.index.columns)
            Index(['build_up_id', 'game_id', 'kick_time', ...])
        """
        if self._index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index file not found: {self.index_path}")
            self._index = pd.read_parquet(self.index_path)
        return self._index

    def load_build_up(self, build_up_id: int) -> pd.DataFrame:
        """Load frame-by-frame tracking data for a specific build-up.

        Args:
            build_up_id: Unique identifier for the build-up window.

        Returns:
            DataFrame with columns: frame, time, player_id, x, y, ball_x, ball_y,
                and additional computed features.

        Raises:
            FileNotFoundError: If frame file doesn't exist for the given build_up_id.

        Example:
            >>> loader = WindowLoader("data/processed/rm_pressing")
            >>> df = loader.load_build_up(42)
            >>> print(df.shape)
            (180, 15)  # 180 frames, 15 columns
        """
        filename = f"build_up_{build_up_id:07d}.parquet"
        file_path = self.frames_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Frame file not found for build_up_id {build_up_id}: {file_path}"
            )

        return pd.read_parquet(file_path)

    def get_metadata(self, build_up_id: int) -> Dict[str, Any]:
        """Retrieve metadata dictionary for a specific build-up.

        Args:
            build_up_id: Unique identifier for the build-up window.

        Returns:
            Dictionary containing all metadata fields from the index row,
            including: game_id, kick_time, ready_time, outcome, etc.

        Raises:
            ValueError: If build_up_id is not found in the index.

        Example:
            >>> loader = WindowLoader("data/processed/rm_pressing")
            >>> meta = loader.get_metadata(42)
            >>> print(meta['outcome'])
            'completed'
        """
        idx = self.index
        row = idx[idx["build_up_id"] == build_up_id]
        if row.empty:
            raise ValueError(f"Build-up ID {build_up_id} not found in index.")
        return row.iloc[0].to_dict()
