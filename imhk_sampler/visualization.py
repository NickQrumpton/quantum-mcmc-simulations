"""
Visualization tools for the IMHK sampler framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from sage.all import *
from sage.structure.element import Vector
from scipy import stats
import seaborn as sns
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global plot configuration
plt.style.use('default')
sns.set_palette("colorblind")

def plot_samples(
    samples: Union[List[Vector], np.ndarray],
    sigma: float,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    dims_to_plot: Optional[List[int]] = None,
    **kwargs
):
    """
    Plot samples based on their dimensionality.
    """
    # Convert to numpy array if needed
    if not isinstance(samples, np.ndarray):
        samples = np.array([list(s) for s in samples])
    
    n_dims = samples.shape[1]
    
    if n_dims == 2:
        return plot_2d_samples(samples, sigma, output_path, title, **kwargs)
    elif n_dims == 3:
        return plot_3d_samples(samples, sigma, output_path, title, **kwargs)
    else:
        # For higher dimensions, plot 2D projections
        return plot_2d_projections(samples, sigma, output_path, title, dims_to_plot, **kwargs)

def plot_2d_samples(
    samples: Union[List[Vector], np.ndarray],
    sigma: float,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_density: bool = True,
    **kwargs
):
    """
    Create 2D scatter plot of samples with optional density overlay.
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array([list(s) for s in samples])
    
    if samples.shape[1] != 2:
        raise ValueError(f"Expected 2D samples, got {samples.shape[1]} dimensions")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Scatter plot
    scatter = ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=30, 
                        edgecolors='black', linewidth=0.5)
    
    # Add density contours if requested
    if show_density:
        try:
            xmin, xmax = samples[:, 0].min() - 3*sigma, samples[:, 0].max() + 3*sigma
            ymin, ymax = samples[:, 1].min() - 3*sigma, samples[:, 1].max() + 3*sigma
            
            # Create grid for theoretical density
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50),
                                np.linspace(ymin, ymax, 50))
            
            # Compute theoretical Gaussian density
            rv = stats.multivariate_normal(mean=[0, 0], cov=[[sigma**2, 0], [0, sigma**2]])
            density = rv.pdf(np.dstack((xx, yy)))
            
            # Add contour lines
            contours = ax.contour(xx, yy, density, levels=5, colors='red', 
                                 alpha=0.8, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=8)
        except Exception as e:
            logger.warning(f"Failed to add density contours: {e}")
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'2D Gaussian Samples (σ={sigma:.3f})')
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_3d_samples(
    samples: Union[List[Vector], np.ndarray],
    sigma: float,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    elevation: float = 30,
    azimuth: float = 45,
    **kwargs
):
    """
    Create 3D scatter plot of samples.
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array([list(s) for s in samples])
    
    if samples.shape[1] != 3:
        raise ValueError(f"Expected 3D samples, got {samples.shape[1]} dimensions")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
                        alpha=0.6, s=20, c='blue', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')  
    ax.set_zlabel('Dimension 3')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'3D Gaussian Samples (σ={sigma:.3f})')
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set equal aspect ratio
    max_range = np.array([samples[:, 0].max()-samples[:, 0].min(),
                         samples[:, 1].max()-samples[:, 1].min(),
                         samples[:, 2].max()-samples[:, 2].min()]).max() / 2.0
    
    mid_x = (samples[:, 0].max()+samples[:, 0].min()) * 0.5
    mid_y = (samples[:, 1].max()+samples[:, 1].min()) * 0.5
    mid_z = (samples[:, 2].max()+samples[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_2d_projections(
    samples: Union[List[Vector], np.ndarray],
    sigma: float,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    dims_to_plot: Optional[List[int]] = None,
    max_projections: int = 6,
    **kwargs
):
    """
    Create 2D projections of high-dimensional samples.
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array([list(s) for s in samples])
    
    n_dims = samples.shape[1]
    
    # Select dimensions to plot
    if dims_to_plot is None:
        if n_dims <= max_projections:
            dims_to_plot = list(range(n_dims))
        else:
            # Plot first few dimensions and some random ones
            dims_to_plot = list(range(min(3, n_dims)))
            if n_dims > 3:
                remaining_dims = list(range(3, n_dims))
                np.random.shuffle(remaining_dims)
                dims_to_plot.extend(remaining_dims[:max_projections-3])
    
    # Create projection pairs
    projection_pairs = []
    for i in range(len(dims_to_plot)):
        for j in range(i+1, len(dims_to_plot)):
            projection_pairs.append((dims_to_plot[i], dims_to_plot[j]))
    
    # Limit number of projections
    if len(projection_pairs) > max_projections:
        projection_pairs = projection_pairs[:max_projections]
    
    # Create subplot grid
    n_plots = len(projection_pairs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each projection
    for idx, (i, j) in enumerate(projection_pairs):
        ax = axes[idx]
        ax.scatter(samples[:, i], samples[:, j], alpha=0.5, s=20)
        ax.set_xlabel(f'Dimension {i}')
        ax.set_ylabel(f'Dimension {j}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(len(projection_pairs), len(axes)):
        axes[idx].set_visible(False)
    
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'{n_dims}D Gaussian Samples - 2D Projections (σ={sigma:.3f})')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_comparison(
    samples_dict: dict,
    sigma: float,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    plot_type: str = '2d',
    **kwargs
):
    """
    Compare multiple sets of samples side by side.
    """
    n_methods = len(samples_dict)
    
    if plot_type == '2d':
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, samples) in enumerate(samples_dict.items()):
            ax = axes[idx]
            
            if not isinstance(samples, np.ndarray):
                samples = np.array([list(s) for s in samples])
            
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=30)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'{method_name}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
    
    elif plot_type == 'density':
        fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
        
        for idx, (method_name, samples) in enumerate(samples_dict.items()):
            if not isinstance(samples, np.ndarray):
                samples = np.array([list(s) for s in samples])
            
            # Scatter plot
            ax_scatter = axes[0, idx]
            ax_scatter.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=30)
            ax_scatter.set_title(f'{method_name} - Samples')
            ax_scatter.grid(True, alpha=0.3)
            ax_scatter.set_aspect('equal')
            
            # Density plot
            ax_density = axes[1, idx]
            try:
                sns.kdeplot(data=samples, x=samples[:,0], y=samples[:,1], 
                           ax=ax_density, levels=5, color='red')
                ax_density.set_title(f'{method_name} - Density')
                ax_density.grid(True, alpha=0.3)
                ax_density.set_aspect('equal')
            except Exception as e:
                logger.warning(f"Failed to create density plot for {method_name}: {e}")
                ax_density.text(0.5, 0.5, "Density plot failed", 
                               transform=ax_density.transAxes,
                               ha='center', va='center')
    
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Sampling Method Comparison (σ={sigma:.3f})')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    annot: bool = True,
    fmt: str = '.2f',
    **kwargs
):
    """
    Create a heatmap visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations if requested
    if annot:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f"{data[i, j]:{fmt}}",
                             ha="center", va="center", color="black")
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def create_diagnostic_report(
    experiment_results: dict,
    output_dir: str,
    title: str = "IMHK Sampler Diagnostic Report"
):
    """
    Create a comprehensive diagnostic report with multiple visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a multi-page PDF report
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = os.path.join(output_dir, 'diagnostic_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, title, ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, f"Dimension: {experiment_results['dimension']}", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.55, f"Basis Type: {experiment_results['basis_type']}", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.5, f"Sigma: {experiment_results['sigma']:.3f}", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.45, f"Samples: {experiment_results['num_samples']}", 
                ha='center', va='center', fontsize=16)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Performance metrics page
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Acceptance rate bar chart
        ax = axes[0, 0]
        methods = ['IMHK']
        acceptance_rates = [experiment_results['imhk_acceptance_rate']]
        if 'klein_time' in experiment_results:
            methods.append('Klein')
            acceptance_rates.append(1.0)  # Klein always accepts
        
        ax.bar(methods, acceptance_rates)
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rates')
        ax.set_ylim(0, 1.1)
        
        # Time comparison
        ax = axes[0, 1]
        times = [experiment_results['imhk_time']]
        if 'klein_time' in experiment_results:
            times.append(experiment_results['klein_time'])
        
        ax.bar(methods[:len(times)], times)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Sampling Time')
        
        # TV distance comparison
        ax = axes[1, 0]
        tv_distances = []
        tv_labels = []
        if experiment_results.get('imhk_tv_distance') is not None:
            tv_distances.append(experiment_results['imhk_tv_distance'])
            tv_labels.append('IMHK')
        if experiment_results.get('klein_tv_distance') is not None:
            tv_distances.append(experiment_results['klein_tv_distance'])
            tv_labels.append('Klein')
        
        if tv_distances:
            ax.bar(tv_labels, tv_distances)
            ax.set_ylabel('Total Variation Distance')
            ax.set_title('TV Distance Comparison')
        else:
            ax.text(0.5, 0.5, 'No TV distance data', ha='center', va='center')
        
        # ESS summary
        ax = axes[1, 1]
        if experiment_results.get('imhk_average_ess') is not None:
            ax.bar(['Average ESS'], [experiment_results['imhk_average_ess']])
            ax.set_ylabel('Effective Sample Size')
            ax.set_title('Average ESS')
        else:
            ax.text(0.5, 0.5, 'No ESS data', ha='center', va='center')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Save additional pages if there are plots in the output directory
        for plot_file in os.listdir(output_dir):
            if plot_file.endswith('.png'):
                img = plt.imread(os.path.join(output_dir, plot_file))
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    logger.info(f"Diagnostic report saved to {pdf_path}")
    return pdf_path