import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sage.all import *

def run_experiment(dim, sigma, num_samples, basis_type='identity', compare_with_klein=True, center=None):
    """
    Run a complete experiment with IMHK sampling and analysis.
    
    Args:
        dim: The dimension of the lattice (2, 3, or 4)
        sigma: The standard deviation of the Gaussian
        num_samples: The number of samples to generate
        basis_type: The type of lattice basis to use ('identity', 'skewed', 'ill-conditioned')
        compare_with_klein: Whether to compare with Klein's algorithm
        center: Center of the Gaussian distribution
        
    Returns:
        A dictionary of results
    """
    from .samplers import imhk_sampler, klein_sampler
    from .diagnostics import plot_trace, plot_autocorrelation, plot_acceptance_trace, compute_autocorrelation, compute_ess
    from .visualization import plot_2d_samples, plot_3d_samples, plot_2d_projections, plot_pca_projection
    from .stats import compute_total_variation_distance, compute_kl_divergence

    # Set up the center
    if center is None:
        center = vector(RR, [0] * dim)
    else:
        center = vector(RR, center)
    
    # Create the lattice basis
    if basis_type == 'identity':
        B = matrix.identity(RR, dim)
    elif basis_type == 'skewed':
        B = matrix.identity(RR, dim)
        B[0, 1] = 1.5  # Add some skew
        if dim >= 3:
            B[0, 2] = 0.5
    elif basis_type == 'ill-conditioned':
        B = matrix.identity(RR, dim)
        # Make the matrix ill-conditioned
        B[0, 0] = 10.0
        B[1, 1] = 0.1
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
    
    experiment_name = f"dim{dim}_sigma{sigma}_{basis_type}"
    if any(c != 0 for c in center):
        experiment_name += f"_center{'_'.join(str(c) for c in center)}"
    
    # Calculate smoothing parameter for reference
    # For an identity basis, the smoothing parameter η_ε(Λ) ≈ sqrt(ln(2n/ε)/π)
    epsilon = 0.01  # A small constant, typically 2^(-n) for some n
    smoothing_param = sqrt(log(2*dim/epsilon)/pi)
    
    print(f"Running experiment: dim={dim}, sigma={sigma}, basis={basis_type}")
    print(f"Smoothing parameter η_{epsilon}(Λ) ≈ {smoothing_param:.4f} for reference")
    print(f"σ/η ratio: {sigma/smoothing_param:.4f}")
    
    # Run IMHK sampler
    burn_in = min(5000, num_samples)  # Use appropriate burn-in
    start_time = time.time()
    imhk_samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
        B, sigma, num_samples, center, burn_in=burn_in)
    imhk_time = time.time() - start_time
    
    # Run Klein sampler for comparison if requested
    if compare_with_klein:
        start_time = time.time()
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_time = time.time() - start_time
    
    # Analyze acceptance rate over time
    plot_acceptance_trace(all_accepts, f"acceptance_trace_{experiment_name}.png")
    
    # Run diagnostics on IMHK samples
    # Trace plots
    plot_trace(imhk_samples, f"trace_imhk_{experiment_name}.png", 
             f"IMHK Sample Trace (σ={sigma}, {basis_type} basis)")
    
    # Autocorrelation
    acf_by_dim = compute_autocorrelation(imhk_samples)
    plot_autocorrelation(acf_by_dim, f"acf_imhk_{experiment_name}.png", 
                       f"IMHK Autocorrelation (σ={sigma}, {basis_type} basis)")
    
    # Effective Sample Size
    ess_values = compute_ess(imhk_samples)
    
    # Visualization
    if dim == 2:
        plot_2d_samples(imhk_samples, sigma, f"samples_imhk_{experiment_name}.png", 
                      B, f"IMHK Samples (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_2d_samples(klein_samples, sigma, f"samples_klein_{experiment_name}.png", 
                          B, f"Klein Samples (σ={sigma}, {basis_type} basis)", center)
    elif dim == 3:
        plot_3d_samples(imhk_samples, sigma, f"samples_imhk_{experiment_name}", 
                      f"IMHK Samples (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_3d_samples(klein_samples, sigma, f"samples_klein_{experiment_name}", 
                          f"Klein Samples (σ={sigma}, {basis_type} basis)", center)
    
    # For higher dimensions, create 2D projections
    if dim >= 3:
        plot_2d_projections(imhk_samples, sigma, f"projections_imhk_{experiment_name}.png", 
                          f"IMHK Projections (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_2d_projections(klein_samples, sigma, f"projections_klein_{experiment_name}.png", 
                              f"Klein Projections (σ={sigma}, {basis_type} basis)", center)
    
    # For all dimensions, create PCA projection to 2D
    plot_pca_projection(imhk_samples, sigma, f"pca_imhk_{experiment_name}.png", 
                      f"IMHK PCA Projection (σ={sigma}, {basis_type} basis)")
    if compare_with_klein:
        plot_pca_projection(klein_samples, sigma, f"pca_klein_{experiment_name}.png", 
                          f"Klein PCA Projection (σ={sigma}, {basis_type} basis)")
    
    # Compute statistical distances
    tv_distance = compute_total_variation_distance(imhk_samples, sigma, B, center)
    
    # Compute KL divergence for small dimensions
    kl_divergence = None
    if dim <= 3:  # Only compute for small dimensions due to computational complexity
        kl_divergence = compute_kl_divergence(imhk_samples, sigma, B, center)
    
    # Compile results
    results = {
        'dimension': dim,
        'sigma': sigma,
        'basis_type': basis_type,
        'center': center,
        'smoothing_parameter': smoothing_param,
        'sigma_smoothing_ratio': sigma/smoothing_param,
        'num_samples': num_samples,
        'burn_in': burn_in,
        'imhk_acceptance_rate': acceptance_rate,
        'imhk_time': imhk_time,
        'imhk_ess': ess_values,
        'imhk_tv_distance': tv_distance,
        'imhk_kl_divergence': kl_divergence
    }
    
    if compare_with_klein:
        results['klein_time'] = klein_time
        
        # Compute TV distance for Klein samples
        klein_tv_distance = compute_total_variation_distance(klein_samples, sigma, B, center)
        results['klein_tv_distance'] = klein_tv_distance
        
        # Compute KL divergence for Klein samples if feasible
        if dim <= 3:
            klein_kl_divergence = compute_kl_divergence(klein_samples, sigma, B, center)
            results['klein_kl_divergence'] = klein_kl_divergence
    
    # Log results
    with open(f"results/logs/experiment_{experiment_name}.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Dimension: {dim}\n")
        f.write(f"Sigma: {sigma}\n")
        f.write(f"Basis type: {basis_type}\n")
        f.write(f"Center: {center}\n")
        f.write(f"Smoothing parameter η_{epsilon}(Λ): {smoothing_param:.6f}\n")
        f.write(f"σ/η ratio: {sigma/smoothing_param:.6f}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Burn-in: {burn_in}\n")
        f.write("\n=== IMHK Results ===\n")
        f.write(f"Acceptance rate: {acceptance_rate:.6f}\n")
        f.write(f"Sampling time: {imhk_time:.6f} seconds\n")
        f.write(f"Effective Sample Size: {ess_values}\n")
        f.write(f"Total Variation distance: {tv_distance:.6f}\n")
        
        if kl_divergence is not None:
            f.write(f"KL divergence: {kl_divergence:.6f}\n")
        
        if compare_with_klein:
            f.write("\n=== Klein Sampler Results ===\n")
            f.write(f"Sampling time: {klein_time:.6f} seconds\n")
            f.write(f"Total Variation distance: {klein_tv_distance:.6f}\n")
            
            if dim <= 3 and 'klein_kl_divergence' in results:
                f.write(f"KL divergence: {results['klein_kl_divergence']:.6f}\n")
            
            f.write("\n=== Comparison ===\n")
            f.write(f"IMHK/Klein time ratio: {imhk_time/klein_time:.6f}\n")
            f.write(f"IMHK/Klein TV distance ratio: {tv_distance/klein_tv_distance:.6f}\n")
            
            if dim <= 3 and 'klein_kl_divergence' in results:
                f.write(f"IMHK/Klein KL divergence ratio: {kl_divergence/results['klein_kl_divergence']:.6f}\n")
    
    # Save all data for later analysis
    with open(f"results/logs/experiment_{experiment_name}.pickle", "wb") as f:
        pickle.dump(results, f)
    
    return results

def parameter_sweep(dimensions=None, sigmas=None, basis_types=None, centers=None, num_samples=1000):
    """
    Perform a parameter sweep across different dimensions, sigmas, basis types, and centers.
    
    Args:
        dimensions: List of dimensions to test (default: [2, 3, 4])
        sigmas: List of sigma values to test (default: [0.5, 1.0, 2.0, 5.0])
        basis_types: List of basis types to test (default: ['identity', 'skewed', 'ill-conditioned'])
        centers: List of centers to test (default: [[0, ..., 0]])
        num_samples: Number of samples to generate for each configuration
        
    Returns:
        A dictionary of results indexed by configuration
    """
    if dimensions is None:
        dimensions = [2, 3, 4]
    
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0, 5.0]
    
    if basis_types is None:
        basis_types = ['identity', 'skewed', 'ill-conditioned']
    
    if centers is None:
        centers = {dim: [vector(RR, [0] * dim)] for dim in dimensions}
    elif isinstance(centers, list):
        # If centers is a list of vectors, assume it applies to all dimensions
        centers = {dim: [vector(RR, c) for c in centers if len(c) == dim] for dim in dimensions}
    
    results = {}
    
    # Create a summary file
    with open("results/logs/parameter_sweep_summary.txt", "w") as summary_file:
        summary_file.write("Parameter Sweep Summary\n")
        summary_file.write("=====================\n\n")
        
        # Loop over all combinations
        for dim in dimensions:
            for sigma in sigmas:
                for basis_type in basis_types:
                    for center in centers.get(dim, [vector(RR, [0] * dim)]):
                        config_key = (dim, sigma, basis_type, tuple(center))
                        
                        # Run the experiment
                        result = run_experiment(
                            dim=dim, 
                            sigma=sigma, 
                            num_samples=num_samples,
                            basis_type=basis_type,
                            compare_with_klein=True,
                            center=center
                        )
                        
                        results[config_key] = result
                        
                        # Log summary information
                        summary_file.write(f"Configuration: dim={dim}, sigma={sigma}, ")
                        summary_file.write(f"basis={basis_type}, center={center}\n")
                        summary_file.write(f"IMHK Acceptance Rate: {result['imhk_acceptance_rate']:.4f}\n")
                        summary_file.write(f"IMHK Total Variation Distance: {result['imhk_tv_distance']:.6f}\n")
                        summary_file.write(f"Klein Total Variation Distance: {result['klein_tv_distance']:.6f}\n")
                        summary_file.write(f"IMHK/Klein TV Ratio: {result['imhk_tv_distance']/result['klein_tv_distance']:.4f}\n")
                        summary_file.write("---\n\n")
    
    # Generate comparative plots
    plot_parameter_sweep_results(results, dimensions, sigmas, basis_types)
    
    return results

def plot_parameter_sweep_results(results, dimensions, sigmas, basis_types):
    """
    Create comparative plots for the parameter sweep results.
    
    Args:
        results: Dictionary of results from parameter_sweep
        dimensions: List of dimensions tested
        sigmas: List of sigma values tested
        basis_types: List of basis types tested
        
    Returns:
        None (saves plots to files)
    """
    # Plot acceptance rate vs. sigma for each dimension and basis type
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_data = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    y_data.append(results[key]['imhk_acceptance_rate'])
            
            if x_data:
                ax.plot(x_data, y_data, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('IMHK Acceptance Rate')
        ax.set_title(f'Acceptance Rate vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/acceptance_vs_sigma_dim{dim}.png')
        plt.close()
    
    # Plot TV distance vs. sigma for each dimension and basis type
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_imhk = []
            y_klein = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    y_imhk.append(results[key]['imhk_tv_distance'])
                    y_klein.append(results[key]['klein_tv_distance'])
            
            if x_data:
                ax.plot(x_data, y_imhk, 'o-', label=f"IMHK {basis_type}")
                ax.plot(x_data, y_klein, 's--', label=f"Klein {basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Total Variation Distance')
        ax.set_title(f'TV Distance vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/tv_distance_vs_sigma_dim{dim}.png')
        plt.close()
    
    # Plot TV distance ratio (IMHK/Klein) vs. sigma
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_ratio = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    ratio = results[key]['imhk_tv_distance'] / results[key]['klein_tv_distance']
                    y_ratio.append(ratio)
            
            if x_data:
                ax.plot(x_data, y_ratio, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('TV Distance Ratio (IMHK/Klein)')
        ax.set_title(f'Quality Improvement Ratio vs. Sigma (Dimension {dim})')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Equal Quality')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/tv_ratio_vs_sigma_dim{dim}.png')
        plt.close()

def compare_convergence_times(results):
    """
    Analyze and compare convergence times across different configurations.
    
    Args:
        results: Dictionary of results from parameter_sweep
        
    Returns:
        None (saves plots to files)
    """
    # Extract dimensions, sigmas, and basis types from results
    dimensions = sorted(set(key[0] for key in results.keys()))
    sigmas = sorted(set(key[1] for key in results.keys()))
    basis_types = sorted(set(key[2] for key in results.keys()))
    
    # Group by dimension
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract ESS-adjusted times
            x_data = []
            y_imhk = []
            y_klein = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    
                    # Compute ESS-adjusted time for IMHK
                    ess_avg = np.mean(results[key]['imhk_ess'])
                    adj_time_imhk = results[key]['imhk_time'] * results[key]['num_samples'] / ess_avg
                    y_imhk.append(adj_time_imhk)
                    
                    # Klein time (no adjustment needed as samples are independent)
                    y_klein.append(results[key]['klein_time'])
            
            if x_data:
                ax.plot(x_data, y_imhk, 'o-', label=f"IMHK {basis_type}")
                ax.plot(x_data, y_klein, 's--', label=f"Klein {basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Time (seconds) per Effective Sample')
        ax.set_title(f'Convergence Time vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/convergence_time_dim{dim}.png')
        plt.close()
        
        # Also plot the ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            x_data = []
            y_ratio = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    
                    # Compute ESS-adjusted time for IMHK
                    ess_avg = np.mean(results[key]['imhk_ess'])
                    adj_time_imhk = results[key]['imhk_time'] * results[key]['num_samples'] / ess_avg
                    
                    # Compute time ratio
                    ratio = adj_time_imhk / results[key]['klein_time']
                    y_ratio.append(ratio)
            
            if x_data:
                ax.plot(x_data, y_ratio, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Time Ratio (IMHK/Klein)')
        ax.set_title(f'Time Overhead Ratio vs. Sigma (Dimension {dim})')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Equal Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/time_ratio_dim{dim}.png')
        plt.close()