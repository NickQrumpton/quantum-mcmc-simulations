import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_2d_samples(samples, sigma, filename, lattice_basis=None, title=None, center=None):
    """
    Create a 2D scatter plot of the samples with optional density contours.
    
    Args:
        samples: List of 2D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        lattice_basis: The lattice basis (for plotting the fundamental domain)
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        
    Returns:
        None (saves plot to file)
    """
    from .utils import discrete_gaussian_pdf

    if center is None:
        center = [0, 0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract x and y coordinates
    x_coords = [float(sample[0]) for sample in samples]
    y_coords = [float(sample[1]) for sample in samples]
    
    # Calculate point density for color mapping
    try:
        xy = np.vstack([x_coords, y_coords])
        z = gaussian_kde(xy)(xy)
        
        # Create a scatter plot with density-based coloring
        scatter = ax.scatter(x_coords, y_coords, c=z, alpha=0.7, 
                           cmap='viridis', s=30, edgecolor='k', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Point Density')
    except:
        # Fallback if KDE fails
        ax.scatter(x_coords, y_coords, alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Add density contours
    x_min, x_max = min(x_coords) - 2, max(x_coords) + 2
    y_min, y_max = min(y_coords) - 2, max(y_coords) + 2
    
    # Create a grid of points
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute the density at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = discrete_gaussian_pdf([X[i, j], Y[i, j]], sigma, center)
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=10, cmap='coolwarm', alpha=0.5)
    
    # Plot the fundamental domain if basis is provided
    if lattice_basis is not None:
        origin = np.array([0, 0])
        v1 = np.array([float(lattice_basis[0, 0]), float(lattice_basis[0, 1])])
        v2 = np.array([float(lattice_basis[1, 0]), float(lattice_basis[1, 1])])
        
        # Plot basis vectors
        ax.arrow(origin[0], origin[1], v1[0], v1[1], head_width=0.2, head_length=0.3, 
               fc='blue', ec='blue', label='Basis Vector 1')
        ax.arrow(origin[0], origin[1], v2[0], v2[1], head_width=0.2, head_length=0.3, 
               fc='green', ec='green', label='Basis Vector 2')
        
        # Plot the parallelogram
        ax.plot([0, v1[0], v1[0]+v2[0], v2[0], 0], 
               [0, v1[1], v1[1]+v2[1], v2[1], 0], 
               'r--', alpha=0.7, label='Fundamental Domain')
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], c='red', s=100, marker='*', 
             label=f'Center {tuple(center)}')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'2D Samples from Discrete Gaussian (σ = {sigma})')
    
    ax.legend(loc='upper right')
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
    ax_histx.hist(x_coords, bins=bins, density=True, alpha=0.7)
    ax_histy.hist(y_coords, bins=bins, density=True, alpha=0.7, orientation='horizontal')
    
    # Add theoretical marginal Gaussian
    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = np.linspace(y_min, y_max, 1000)
    
    x_density = [discrete_gaussian_pdf(x, sigma, center[0]) for x in x_vals]
    x_density = np.array(x_density) / sum(x_density) * len(x_vals) / (x_max-x_min)
    
    y_density = [discrete_gaussian_pdf(y, sigma, center[1]) for y in y_vals]
    y_density = np.array(y_density) / sum(y_density) * len(y_vals) / (y_max-y_min)
    
    ax_histx.plot(x_vals, x_density, 'r-', alpha=0.7)
    ax_histy.plot(y_density, y_vals, 'r-', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_3d_samples(samples, sigma, filename, title=None, center=None):
    """
    Create a 3D scatter plot of the samples.
    
    Args:
        samples: List of 3D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        
    Returns:
        None (saves plot to file)
    """
    if center is None:
        center = [0, 0, 0]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, and z coordinates
    x_coords = [float(sample[0]) for sample in samples]
    y_coords = [float(sample[1]) for sample in samples]
    z_coords = [float(sample[2]) for sample in samples]
    
    # Create a scatter plot
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                       c=z_coords, cmap='viridis', 
                       alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], [center[2]], 
             c='red', s=100, marker='*')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'3D Samples from Discrete Gaussian (σ = {sigma})')
    
    plt.colorbar(scatter, ax=ax, label='Dimension 3 Value')
    
    # Save multiple views for 3D visualization
    views = [(30, 30), (0, 0), (0, 90), (90, 0)]
    for i, (elev, azim) in enumerate(views):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f'results/plots/{filename}_view{i+1}.png')
    
    plt.close()

def plot_2d_projections(samples, sigma, filename, title=None, center=None):
    """
    Create 2D projections of higher-dimensional samples.
    
    Args:
        samples: List of samples (3D or 4D)
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    if n_dims < 3:
        print("Warning: 2D Projections are only meaningful for dimensions >= 3")
        return
    
    if center is None:
        center = [0] * n_dims
    
    # Create all pairs of dimensions
    pairs = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)]
    n_pairs = len(pairs)
    
    fig = plt.figure(figsize=(5 * n_pairs, 15))
    
    # Calculate point density for each projection
    for idx, (dim1, dim2) in enumerate(pairs):
        ax = fig.add_subplot(3, n_pairs//3 + (1 if n_pairs % 3 else 0), idx+1)
        
        # Extract the coordinates for this pair of dimensions
        x_coords = [float(sample[dim1]) for sample in samples]
        y_coords = [float(sample[dim2]) for sample in samples]
        
        # Calculate point density for coloring
        try:
            xy = np.vstack([x_coords, y_coords])
            z = gaussian_kde(xy)(xy)
            
            # Create a scatter plot with density-based coloring
            scatter = ax.scatter(x_coords, y_coords, c=z, 
                               cmap='viridis', alpha=0.7, s=30, 
                               edgecolor='k', linewidth=0.5)
        except:
            # Fallback if KDE fails
            ax.scatter(x_coords, y_coords, alpha=0.7, s=30, 
                     edgecolor='k', linewidth=0.5)
        
        # Mark the center for this projection
        ax.scatter([center[dim1]], [center[dim2]], 
                 c='red', s=100, marker='*')
        
        ax.set_xlabel(f'Dimension {dim1+1}')
        ax.set_ylabel(f'Dimension {dim2+1}')
        ax.set_title(f'Proj: Dim {dim1+1} vs Dim {dim2+1}')
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'{n_dims}D Samples Projected to 2D (σ = {sigma})', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_pca_projection(samples, sigma, filename, title=None):
    """
    Create a PCA projection of higher-dimensional samples to 2D.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    if n_dims < 3:
        print("Warning: PCA projection to 2D is only meaningful for dimensions >= 3")
        return
    
    # Convert samples to numpy array
    samples_array = np.array([[float(x) for x in sample] for sample in samples])
    
    # Apply PCA
    pca = PCA(n_components=2)
    samples_pca = pca.fit_transform(samples_array)
    
    # Plot the projection
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate point density for coloring
    try:
        xy = np.vstack([samples_pca[:, 0], samples_pca[:, 1]])
        z = gaussian_kde(xy)(xy)
        
        # Create a scatter plot with density-based coloring
        scatter = ax.scatter(samples_pca[:, 0], samples_pca[:, 1], c=z, 
                           cmap='viridis', alpha=0.7, s=30, 
                           edgecolor='k', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Point Density')
    except:
        # Fallback if KDE fails
        ax.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.7, s=30, 
                 edgecolor='k', linewidth=0.5)
    
    # Add explained variance ratio
    ax.text(0.02, 0.98, 
           f'Explained variance:\nPC1: {pca.explained_variance_ratio_[0]:.3f}\n'
           f'PC2: {pca.explained_variance_ratio_[1]:.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'PCA Projection to 2D (σ = {sigma}, {n_dims}D → 2D)')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()