"""Plotting utilities for analysis validation and model diagnostics.

This module provides visualization functions for model evaluation metrics and
zone visualization from GMM fitting.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture


def plot_reconstruction_error(errors: List[float], out_path: Path) -> None:
    """Plot NMF reconstruction error across different parameter settings.

    Creates a line plot showing reconstruction error trends, useful for
    determining optimal NMF component counts.

    Args:
        errors: List of reconstruction error values.
        out_path: Output file path for the saved plot.

    Returns:
        None. Saves plot to out_path.

    Example:
        >>> errors = [0.45, 0.32, 0.28, 0.27, 0.26]
        >>> plot_reconstruction_error(errors, Path("error_plot.png"))
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors)), errors, marker='o')
    plt.xlabel("Parameter Index")
    plt.ylabel("Reconstruction Error")
    plt.title("NMF Reconstruction Error")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_gmm_zones(gmm: GaussianMixture, out_path: Path,
                   data: Optional[pd.DataFrame] = None) -> None:
    """Visualize GMM zone centroids on a pitch coordinate system.

    Creates a scatter plot showing the mean positions of each GMM zone,
    optionally overlaying the original data points.

    Args:
        gmm: Fitted GaussianMixture model with means_ attribute.
        out_path: Output file path for the saved plot.
        data: Optional DataFrame with 'x' and 'y' columns to overlay data points.

    Returns:
        None. Saves plot to out_path.

    Raises:
        ValueError: If GMM model is not fitted (no means_ attribute).

    Example:
        >>> from sklearn.mixture import GaussianMixture
        >>> gmm = GaussianMixture(n_components=8).fit(positions)
        >>> plot_gmm_zones(gmm, Path("zones.png"))
    """
    if not hasattr(gmm, 'means_'):
        raise ValueError("GMM model must be fitted before plotting")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data points if provided
    if data is not None:
        ax.scatter(data['x'], data['y'], alpha=0.1, s=1, c='gray', label='Data')

    # Plot GMM zone centroids
    means = gmm.means_
    ax.scatter(means[:, 0], means[:, 1], c='red', s=200, marker='X',
               edgecolors='black', linewidths=2, label='Zone Centroids', zorder=10)

    # Annotate zone numbers
    for i, (x, y) in enumerate(means):
        ax.annotate(f'Z{i}', (x, y), fontsize=12, ha='center', va='center')

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(f"GMM Pressing Zones (n={gmm.n_components})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
