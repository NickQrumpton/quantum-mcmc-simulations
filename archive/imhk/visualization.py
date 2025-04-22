#!/usr/bin/env sage
"""
Visualization Functions for IMHK Algorithm
------------------------------------------
This module contains plotting functions for visualizing IMHK results.

EXAMPLES::

    >>> # Most visualization functions require plotting backends
    >>> # Check that set_publication_style() runs without error
    >>> set_publication_style()  # doctest: +SKIP
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any

from .core import discrete_gaussian_pdf
from imhk import logger

def set_publication_style():
    """
    Configure Matplotlib for publication-quality plots.

    Uses a built-in Matplotlib style ('ggplot') for compatibility with SageMath.
    """
    try:
        plt.style.use('ggplot')
    except Exception as e:
        logger.warning(f"Could not set plot style 'ggplot': {e}")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (5.5, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def plot_2d_samples(samples, sigma, filename, lattice_basis, title, center=None):
    """
    Create a 2D scatter plot with Gaussian contours and basis vectors.

    Args:
        samples: List of 2D sample vectors.
        sigma: Standard deviation of the discrete Gaussian.
        filename: File path to save the plot.
        lattice_basis: Lattice basis matrix.
        title: Title of the plot.
        center: Center of the Gaussian. Defaults to the origin.
    """
    try:
        set_publication_style()
        samples_np = np.array([np.array(sample) for sample in samples])
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot samples
        ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10)

        # Plot basis vectors from the origin or center
        origin = [0, 0] if center is None else [center[0], center[1]]
        for i in range(lattice_basis.nrows()):
            vec = lattice_basis[i]
            ax.arrow(origin[0], origin[1], vec[0], vec[1],
                     head_width=0.1, head_length=0.2,
                     fc='red', ec='red', linewidth=2)

        # Set up grid for contours
        center_np = np.array([0, 0]) if center is None else np.array(center)
        x_min, x_max = samples_np[:, 0].min() - sigma, samples_np[:, 0].max() + sigma
        y_min, y_max = samples_np[:, 1].min() - sigma, samples_np[:, 1].max() + sigma
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pt = np.array([X[i, j], Y[i, j]])
                Z[i, j] = discrete_gaussian_pdf(pt, sigma, center_np)

        # Plot contours
        contour = ax.contour(X, Y, Z, cmap='viridis', levels=10)
        plt.colorbar(contour, ax=ax, label="Density")

        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect('equal')
        plt.tight_layout()

        # Ensure output directory exists
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

        plt.savefig(filename)
        plt.close()
        logger.info(f"Plot saved to {filename}")

    except Exception as e:
        logger.error(f"Error creating 2D sample plot: {e}")