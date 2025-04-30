import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Union, Optional, Tuple, Any
from sage.structure.element import Vector
from pathlib import Path
import os
from functools import partial

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def get_results_path():
    """Get absolute path to results directory."""
    from pathlib import Path
    
    # Get the absolute path of the current file
    current_file = Path(__file__).resolve()
    
    # Get the project root
    project_root = current_file.parent.parent
    
    # Create and return the results path
    results_path = project_root / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    return results_path


def _to_numpy(samples: List[Union[Vector, np.ndarray, List]]) -> np.ndarray:
    """
    Convert SageMath vectors or lists to a NumPy array.
    
    Args:
        samples: List of vectors to convert
        
    Returns:
        NumPy array of the samples
    """
    if not samples:
        raise ValueError("Empty samples list provided")
    
    # Create an empty array based on the sample shape
    sample = samples[0]
    if isinstance(sample, Vector):
        # SageMath vector
        n_dims = len(sample)
        result = np.zeros((len(samples), n_dims))
        for i, s in enumerate(samples):
            for j in range(n_dims):
                result[i, j] = float(s[j])
    elif isinstance(sample, (list, tuple)):
        # List or tuple
        result = np.array(samples, dtype=float)
    elif isinstance(sample, np.ndarray):
        # Already a NumPy array, just stack them
        result = np.stack(samples) if len(samples) > 1 else np.array(samples[0], dtype=float)
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")
    
    return result


def plot_2d_samples(samples: List[Union[Vector, np.ndarray, List]], 
                   sigma: float, 
                   filename: str, 
                   lattice_basis: Optional[Union[np.ndarray, List]] = None, 
                   title: Optional[str] = None, 
                   center: Optional[Union[List, np.ndarray, Vector]] = None,
                   discrete_gaussian_pdf: Optional[callable] = None) -> None:
    """
    Create a 2D scatter plot of the samples with density contours and marginal distributions.
    
    Lattice-based Cryptography Context:
    This visualization helps verify that the discrete Gaussian sampler is correctly 
    generating samples according to the target distribution, which is crucial for:
    - Validating security assumptions in lattice-based cryptographic schemes
    - Assessing the quality of trapdoor sampling for signatures
    - Analyzing the coverage of the fundamental domain of the lattice
    
    The plot includes:
    - Density-colored scatter plot of sample points
    - Theoretical Gaussian density contours
    - Marginal distributions along each axis
    - Lattice fundamental domain (if basis provided)
    
    Args:
        samples: List of 2D samples (SageMath vectors, lists, or NumPy arrays)
        sigma: The standard deviation of the Gaussian (must be positive)
        filename: Filename to save the plot (without path)
        lattice_basis: The lattice basis for plotting the fundamental domain 
                      (2×2 matrix where rows are basis vectors)
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        discrete_gaussian_pdf: Function to compute discrete Gaussian PDF
                             (if None, will be imported from utils)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions mismatch
        TypeError: If input types are not compatible
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    
    # Import discrete_gaussian_pdf if not provided
    if discrete_gaussian_pdf is None:
        from imhk_sampler.utils import discrete_gaussian_pdf
    
    # Validate and convert samples
    samples_np = _to_numpy(samples)
    
    # Check dimensionality
    if samples_np.shape[1] != 2:
        raise ValueError(f"Expected 2D samples, got {samples_np.shape[1]}D")
    
    # Handle center
    if center is None:
        center = np.array([0, 0])
    else:
        center = np.array(center, dtype=float)
        if len(center) != 2:
            raise ValueError(f"Center must be 2D, got {len(center)}D")
    
    # Ensure output directory exists
    plot_dir = get_results_path() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with higher quality settings
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract x and y coordinates
    x_coords = samples_np[:, 0]
    y_coords = samples_np[:, 1]
    
    # Calculate point density for color mapping
    # For very large datasets, downsample for KDE calculation
    max_kde_points = 5000
    if len(samples_np) > max_kde_points:
        # Random selection of points for KDE
        idx = np.random.choice(len(samples_np), max_kde_points, replace=False)
        kde_points = samples_np[idx]
        xy_kde = np.vstack([kde_points[:, 0], kde_points[:, 1]])
    else:
        xy_kde = np.vstack([x_coords, y_coords])
    
    try:
        # Compute KDE for coloring
        kde = gaussian_kde(xy_kde)
        z = kde(np.vstack([x_coords, y_coords]))
        
        # Create a scatter plot with density-based coloring
        scatter = ax.scatter(x_coords, y_coords, c=z, alpha=0.7, 
                           cmap='viridis', s=30, edgecolor='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, label='Point Density')
        cbar.ax.tick_params(labelsize=10)
    except Exception as e:
        # Fallback if KDE fails
        print(f"KDE calculation failed: {e}. Using uniform coloring.")
        ax.scatter(x_coords, y_coords, alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Add density contours - vectorized computation
    margin = 2.0  # Add margin around data points
    x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
    y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
    
    # Create a grid of points
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Vectorize the computation using a numpy array of points
    points = np.column_stack((X.flatten(), Y.flatten()))
    Z_flat = np.zeros(len(points))
    
    # Vectorized calculation of discrete Gaussian PDF
    for i, point in enumerate(points):
        Z_flat[i] = discrete_gaussian_pdf(point, sigma, center)
    
    # Reshape Z back to grid shape
    Z = Z_flat.reshape(X.shape)
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=10, cmap='coolwarm', alpha=0.5)
    
    # Plot the fundamental domain if basis is provided
    if lattice_basis is not None:
        try:
            lattice_basis = np.array(lattice_basis, dtype=float)
            if lattice_basis.shape != (2, 2):
                raise ValueError(f"Lattice basis must be 2×2, got {lattice_basis.shape}")
                
            origin = np.array([0, 0])
            v1 = lattice_basis[0]
            v2 = lattice_basis[1]
            
            # Plot basis vectors
            ax.arrow(origin[0], origin[1], v1[0], v1[1], head_width=0.2, head_length=0.3, 
                   fc='blue', ec='blue', label='Basis Vector 1')
            ax.arrow(origin[0], origin[1], v2[0], v2[1], head_width=0.2, head_length=0.3, 
                   fc='green', ec='green', label='Basis Vector 2')
            
            # Plot the parallelogram
            ax.plot([0, v1[0], v1[0]+v2[0], v2[0], 0], 
                   [0, v1[1], v1[1]+v2[1], v2[1], 0], 
                   'r--', alpha=0.7, label='Fundamental Domain')
            
            # Adding lattice points within view range
            # This helps visualize the discrete nature of the lattice
            lattice_range = 3  # Number of basis vectors to show in each direction
            lattice_points = []
            for i in range(-lattice_range, lattice_range + 1):
                for j in range(-lattice_range, lattice_range + 1):
                    point = i * v1 + j * v2
                    lattice_points.append(point)
            
            lattice_points = np.array(lattice_points)
            # Plot only points within axis limits with small markers
            in_range = ((lattice_points[:, 0] >= x_min) & (lattice_points[:, 0] <= x_max) &
                        (lattice_points[:, 1] >= y_min) & (lattice_points[:, 1] <= y_max))
            if any(in_range):
                ax.scatter(lattice_points[in_range, 0], lattice_points[in_range, 1],
                         color='black', s=15, alpha=0.8, marker='x', label='Lattice Points')
        
        except Exception as e:
            print(f"Error plotting lattice basis: {e}")
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], c='red', s=100, marker='*', 
             label=f'Center ({center[0]:.1f}, {center[1]:.1f})')
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'2D Samples from Discrete Gaussian (σ = {sigma:.2f})', fontsize=14)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add histogram subplots for marginal distributions
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
    
    # Make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    # Add histograms
    bins = min(50, int(np.sqrt(len(samples))))
    ax_histx.hist(x_coords, bins=bins, density=True, alpha=0.7, color='navy')
    ax_histy.hist(y_coords, bins=bins, density=True, alpha=0.7, orientation='horizontal', color='navy')
    
    # Add theoretical marginal Gaussian - vectorized computation
    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = np.linspace(y_min, y_max, 1000)
    
    # Vectorized 1D Gaussian calculation
    x_density = np.array([discrete_gaussian_pdf([x, center[1]], sigma, center) for x in x_vals])
    x_density = x_density / np.sum(x_density) * len(x_vals) / (x_max - x_min)
    
    y_density = np.array([discrete_gaussian_pdf([center[0], y], sigma, center) for y in y_vals])
    y_density = y_density / np.sum(y_density) * len(y_vals) / (y_max - y_min)
    
    ax_histx.plot(x_vals, x_density, 'r-', alpha=0.7, linewidth=2)
    ax_histy.plot(y_density, y_vals, 'r-', alpha=0.7, linewidth=2)
    
    # Set axis labels for marginal distributions
    ax_histx.set_ylabel('Density', fontsize=10)
    ax_histy.set_xlabel('Density', fontsize=10)
    
    plt.tight_layout()
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)


def plot_3d_samples(samples: List[Union[Vector, np.ndarray, List]], 
                   sigma: float, 
                   filename: str, 
                   title: Optional[str] = None, 
                   center: Optional[Union[List, np.ndarray, Vector]] = None,
                   single_view: bool = False) -> None:
    """
    Create a 3D scatter plot of the samples.
    
    Lattice-based Cryptography Context:
    3D visualizations help understand the distribution of samples in higher dimensions, 
    which is crucial for analyzing:
    - The quality of discrete Gaussian sampling in higher dimensions
    - The geometric distribution of lattice points relevant to security parameters
    - Potential clustering or biases in the sampling process
    
    Args:
        samples: List of 3D samples (SageMath vectors, lists, or NumPy arrays)
        sigma: The standard deviation of the Gaussian (must be positive)
        filename: Filename to save the plot (without path)
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        single_view: If True, saves only a single view; otherwise saves multiple views
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions mismatch
        TypeError: If input types are not compatible
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    
    # Validate and convert samples
    samples_np = _to_numpy(samples)
    
    # Check dimensionality
    if samples_np.shape[1] != 3:
        raise ValueError(f"Expected 3D samples, got {samples_np.shape[1]}D")
    
    # Handle center
    if center is None:
        center = np.array([0, 0, 0])
    else:
        center = np.array(center, dtype=float)
        if len(center) != 3:
            raise ValueError(f"Center must be 3D, got {len(center)}D")
    
    # Ensure output directory exists
    plot_dir = get_results_path() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with higher quality settings
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x_coords = samples_np[:, 0]
    y_coords = samples_np[:, 1]
    z_coords = samples_np[:, 2]
    
    # Calculate distances from center for coloring
    distances = np.sqrt(np.sum((samples_np - center)**2, axis=1))
    
    # Create a scatter plot with distance-based coloring
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                       c=distances, cmap='viridis', 
                       alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], [center[2]], 
             c='red', s=150, marker='*', label='Center')
    
    # Improve axis labels and font sizes
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_zlabel('Dimension 3', fontsize=12)
    
    # Equal aspect ratio for better visualization
    # This is crucial for lattice visualization
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]),
        np.max([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'3D Samples from Discrete Gaussian (σ = {sigma:.2f})', fontsize=14)
    
    # Add colorbar and legend
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, 
                      label='Distance from Center')
    cbar.ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    
    # Either save a single view or multiple views
    if single_view:
        # Default view
        ax.view_init(elev=30, azim=30)
        plt.tight_layout()
        try:
            plt.savefig(plot_dir / filename, dpi=300)
            print(f"Plot saved to {plot_dir / filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        # Save multiple views for 3D visualization
        views = [(30, 30), (0, 0), (0, 90), (90, 0)]
        for i, (elev, azim) in enumerate(views):
            ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            view_filename = f"{os.path.splitext(filename)[0]}_view{i+1}{os.path.splitext(filename)[1]}"
            try:
                plt.savefig(plot_dir / view_filename, dpi=300)
                print(f"Plot saved to {plot_dir / view_filename}")
            except Exception as e:
                print(f"Error saving plot {i+1}: {e}")
    
    plt.close(fig)


def plot_2d_projections(samples: List[Union[Vector, np.ndarray, List]], 
                       sigma: float, 
                       filename: str, 
                       title: Optional[str] = None, 
                       center: Optional[Union[List, np.ndarray, Vector]] = None) -> None:
    """
    Create 2D projections of higher-dimensional samples.
    
    Lattice-based Cryptography Context:
    2D projections provide critical insights into high-dimensional lattice structures by:
    - Revealing correlations between dimensions in multivariate Gaussian distributions
    - Enabling visualization of potential weaknesses in sampling algorithms
    - Helping detect non-uniformity that might impact security properties
    - Verifying that the high-dimensional sampler maintains expected distribution properties
      across all projection planes
    
    Args:
        samples: List of samples in 3+ dimensions (SageMath vectors, lists, or NumPy arrays)
        sigma: The standard deviation of the Gaussian (must be positive)
        filename: Filename to save the plot (without path)
        title: Optional title for the plot
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions are too few
        TypeError: If input types are not compatible
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    
    # Validate and convert samples
    samples_np = _to_numpy(samples)
    n_dims = samples_np.shape[1]
    
    # Check dimensionality
    if n_dims < 3:
        raise ValueError("2D Projections are only meaningful for dimensions >= 3")
    
    # Handle center
    if center is None:
        center = np.zeros(n_dims)
    else:
        center = np.array(center, dtype=float)
        if len(center) != n_dims:
            raise ValueError(f"Center dimensions ({len(center)}) must match sample dimensions ({n_dims})")
    
    # Ensure output directory exists
    plot_dir = get_results_path() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all pairs of dimensions
    pairs = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)]
    n_pairs = len(pairs)
    
    # Determine grid layout based on number of pairs
    if n_pairs <= 3:
        n_cols = n_pairs
        n_rows = 1
    else:
        n_cols = 3
        n_rows = (n_pairs + 2) // 3  # Ceiling division
    
    # Create figure with higher quality settings
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    # Downsample for KDE if needed
    max_kde_points = 5000
    if len(samples_np) > max_kde_points:
        kde_indices = np.random.choice(len(samples_np), max_kde_points, replace=False)
        kde_samples = samples_np[kde_indices]
    else:
        kde_samples = samples_np
    
    # Calculate point density for each projection
    for idx, (dim1, dim2) in enumerate(pairs):
        ax = fig.add_subplot(n_rows, n_cols, idx+1)
        
        # Extract the coordinates for this pair of dimensions
        x_coords = samples_np[:, dim1]
        y_coords = samples_np[:, dim2]
        
        # Calculate point density for coloring
        try:
            xy_kde = np.vstack([kde_samples[:, dim1], kde_samples[:, dim2]])
            kde = gaussian_kde(xy_kde)
            z = kde(np.vstack([x_coords, y_coords]))
            
            # Create a scatter plot with density-based coloring
            scatter = ax.scatter(x_coords, y_coords, c=z, 
                               cmap='viridis', alpha=0.7, s=30, 
                               edgecolor='k', linewidth=0.5)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label('Density', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            # Fallback if KDE fails
            print(f"KDE calculation failed for dimensions {dim1+1} and {dim2+1}: {e}")
            ax.scatter(x_coords, y_coords, alpha=0.7, s=30, 
                     edgecolor='k', linewidth=0.5)
        
        # Mark the center for this projection
        ax.scatter([center[dim1]], [center[dim2]], 
                 c='red', s=100, marker='*', label='Center')
        
        # Improve labels and styling
        ax.set_xlabel(f'Dimension {dim1+1}', fontsize=11)
        ax.set_ylabel(f'Dimension {dim2+1}', fontsize=11)
        ax.set_title(f'Projection: Dim {dim1+1} vs Dim {dim2+1}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Calculate correlation coefficient
        corr = np.corrcoef(x_coords, y_coords)[0, 1]
        # Add correlation information
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'{n_dims}D Samples Projected to 2D (σ = {sigma:.2f})', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)


def plot_pca_projection(samples: List[Union[Vector, np.ndarray, List]], 
                       sigma: float, 
                       filename: str, 
                       title: Optional[str] = None,
                       center: Optional[Union[List, np.ndarray, Vector]] = None) -> None:
    """
    Create a PCA projection of higher-dimensional samples to 2D.
    
    Lattice-based Cryptography Context:
    PCA projections are valuable for analyzing high-dimensional lattice samples by:
    - Revealing the primary axes of variance in the sampling distribution
    - Identifying potential weaknesses in sampling algorithms
    - Providing intuition about the behavior of cryptographic constructions
      in high-dimensional lattices
    - Showing the effective dimensionality of the sample distribution, which
      relates to the entropy and security of cryptographic schemes
    
    Args:
        samples: List of samples (3+ dimensions) (SageMath vectors, lists, or NumPy arrays)
        sigma: The standard deviation of the Gaussian (must be positive)
        filename: Filename to save the plot (without path)
        title: Optional title for the plot
        center: Center of the Gaussian distribution (used for annotation)
        
    Returns:
        None (saves plot to file)
        
    Raises:
        ValueError: If samples is empty, sigma is not positive, or dimensions are too few
        TypeError: If input types are not compatible
    """
    # Input validation
    if not samples:
        raise ValueError("Empty samples list provided")
    if sigma <= 0:
        raise ValueError("Standard deviation (sigma) must be positive")
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    
    # Validate and convert samples
    samples_np = _to_numpy(samples)
    n_dims = samples_np.shape[1]
    
    # Check dimensionality
    if n_dims < 3:
        raise ValueError("PCA projection to 2D is only meaningful for dimensions >= 3")
    
    # Handle center
    if center is None:
        center = np.zeros(n_dims)
    else:
        center = np.array(center, dtype=float)
        if len(center) != n_dims:
            raise ValueError(f"Center dimensions ({len(center)}) must match sample dimensions ({n_dims})")
    
    # Ensure output directory exists
    plot_dir = get_results_path() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply PCA
    pca = PCA(n_components=2)
    samples_pca = pca.fit_transform(samples_np)
    
    # Project the center point
    center_pca = pca.transform(center.reshape(1, -1))[0]
    
    # Create figure with higher quality settings
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate point density for coloring
    try:
        # For very large datasets, downsample for KDE calculation
        max_kde_points = 5000
        if len(samples_pca) > max_kde_points:
            idx = np.random.choice(len(samples_pca), max_kde_points, replace=False)
            kde_points = samples_pca[idx]
            xy_kde = np.vstack([kde_points[:, 0], kde_points[:, 1]])
        else:
            xy_kde = np.vstack([samples_pca[:, 0], samples_pca[:, 1]])
        
        kde = gaussian_kde(xy_kde)
        z = kde(np.vstack([samples_pca[:, 0], samples_pca[:, 1]]))
        
        # Create a scatter plot with density-based coloring
        scatter = ax.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.7, s=30, 
                 edgecolor='k', linewidth=0.5)
    
    # Mark the projected center
    ax.scatter([center_pca[0]], [center_pca[1]], c='red', s=100, marker='*', 
             label=f'Projected Center')
    
    # Add explained variance ratio
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.sum(variance_explained)
    
    # Add detailed annotation with PCA statistics
    ax.text(0.02, 0.98, 
           f'Explained variance:\n'
           f'PC1: {variance_explained[0]:.3f} ({variance_explained[0]*100:.1f}%)\n'
           f'PC2: {variance_explained[1]:.3f} ({variance_explained[1]*100:.1f}%)\n'
           f'Total: {cumulative_variance:.3f} ({cumulative_variance*100:.1f}%)',
           transform=ax.transAxes, verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add eigenvector directions as arrows (showing principal components)
    scaling_factor = 3 * np.sqrt(pca.explained_variance_)
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * scaling_factor[i]  # Scale by explained variance
        ax.arrow(center_pca[0], center_pca[1], comp[0], comp[1], 
               head_width=0.2, head_length=0.3, fc=f'C{i}', ec=f'C{i}',
               label=f'PC{i+1} ({variance_explained[i]:.3f})')
    
    # Improve axis labels and appearance
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    
    # Equal aspect ratio for unbiased visual comparison
    ax.set_aspect('equal')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'PCA Projection: {n_dims}D → 2D (σ = {sigma:.2f})', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add the top 5 feature contributions to PC1 and PC2
    feature_importance = np.abs(pca.components_)
    top_features_pc1 = np.argsort(feature_importance[0])[::-1][:5]
    top_features_pc2 = np.argsort(feature_importance[1])[::-1][:5]
    
    feat_text = "Top dimension contributions:\n"
    feat_text += "PC1: " + ", ".join([f"Dim{i+1} ({feature_importance[0][i]:.2f})" 
                                     for i in top_features_pc1]) + "\n"
    feat_text += "PC2: " + ", ".join([f"Dim{i+1} ({feature_importance[0][i]:.2f})" 
                                     for i in top_features_pc2])
    
    ax.text(0.02, 0.02, feat_text, transform=ax.transAxes, 
           verticalalignment='bottom', horizontalalignment='left', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    try:
        plt.savefig(plot_dir / filename, dpi=300)
        print(f"Plot saved to {plot_dir / filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)_pca[:, 1], c=z, 
                           cmap='viridis', alpha=0.7, s=30, 
                           edgecolor='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, label='Point Density')
        cbar.ax.tick_params(labelsize=10)
    except Exception as e:
        # Fallback if KDE fails
        print(f"KDE calculation failed: {e}. Using uniform coloring.")
        ax.scatter(samples_pca[:, 0], samples
