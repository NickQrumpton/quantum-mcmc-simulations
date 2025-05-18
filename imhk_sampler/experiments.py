"""
Core experimental framework for lattice-based discrete Gaussian samplers with cryptographic applications.

This module provides comprehensive tools for running experiments with discrete Gaussian samplers
over lattices, with a focus on cryptographically relevant parameter ranges and metrics. The
experiments are designed to validate the IMHK sampler against Klein's algorithm and establish
its advantages in higher dimensions and challenging lattice structures.
"""

# Control sklearn usage
USE_SKLEARN = False

from sage.all import *
import numpy as np
import time
import pickle
import json
import os
import sys
import logging

# Import create_lattice_basis from utils
from .utils import create_lattice_basis

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure matplotlib to avoid backend issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure constants
DEFAULT_RESULTS_DIR = "results/"
DEFAULT_PLOTS_DIR = "results/plots/"
DEFAULT_DATA_DIR = "results/data/"
DEFAULT_LOGS_DIR = "results/logs/"

# Module level directories
results_dir = DEFAULT_RESULTS_DIR
plots_dir = DEFAULT_PLOTS_DIR
data_dir = DEFAULT_DATA_DIR
logs_dir = DEFAULT_LOGS_DIR

def init_directories(base_dir=None):
    """Initialize output directories for results, plots, data, and logs."""
    global results_dir, plots_dir, data_dir, logs_dir
    
    if base_dir:
        results_dir = base_dir
    else:
        results_dir = DEFAULT_RESULTS_DIR
    
    plots_dir = os.path.join(results_dir, "plots")
    data_dir = os.path.join(results_dir, "data")
    logs_dir = os.path.join(results_dir, "logs")
    
    # Create directories if they don't exist
    for dir_path in [results_dir, plots_dir, data_dir, logs_dir]:
        os.makedirs(dir_path, exist_ok=True)

# Initialize directories
init_directories()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lattice_experiments')

# Function moved to utils.py

def validate_basis_type(basis_type):
    """Validate that a basis type is supported."""
    supported_types = ['identity', 'ill-conditioned', 'skewed', 'q-ary'] 
    return basis_type in supported_types

def run_experiment(dim, sigma, num_samples, basis_type='identity', compare_with_klein=True, center=None,
                  plot_results=True, save_results=True):
    """
    Run a complete discrete Gaussian sampling experiment comparing IMHK and Klein samplers.
    
    This function serves as the primary experimental framework for evaluating
    discrete Gaussian samplers in cryptographically relevant scenarios.
    
    Cryptographic Context:
    The discrete Gaussian distribution is fundamental to lattice-based cryptography,
    appearing in schemes like:
    - Learning With Errors (LWE) and Ring-LWE
    - NTRU-based encryption
    - Hash-and-sign signatures (FALCON)
    - IBE and ABE constructions
    
    The ratio σ/η (Gaussian parameter to smoothing parameter) is crucial:
    - σ/η > 1: Ensures statistical properties required for security
    - Higher ratios generally provide better security but may impact efficiency
    
    Args:
        dim: The dimension of the lattice (must be ≥ 2)
        sigma: The standard deviation of the Gaussian (must be positive)
        num_samples: The number of samples to generate (must be positive)
        basis_type: The type of lattice basis to use (default: 'identity')
        compare_with_klein: Whether to compare with Klein's algorithm (default: True)
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        A dictionary containing the experiment results
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Import required functions
    from samplers import imhk_sampler, klein_sampler, imhk_sampler_wrapper, klein_sampler_wrapper
    from diagnostics import plot_trace, plot_autocorrelation, plot_acceptance_trace, compute_autocorrelation, compute_ess
    from visualization import plot_2d_samples, plot_3d_samples, plot_2d_projections
    from stats import compute_total_variation_distance, compute_kl_divergence
    
    # Input validation
    if not isinstance(dim, (int, Integer)) or dim < 2:
        raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    
    if not isinstance(sigma, (float, RealNumber)) or sigma <= 0:
        raise ValueError(f"Standard deviation (sigma) must be positive, got {sigma}")
    
    if not isinstance(num_samples, (int, Integer)) or num_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {num_samples}")
    
    if not validate_basis_type(basis_type):
        raise ValueError(f"Unknown basis type: {basis_type}. Valid options are 'identity', 'skewed', 'ill-conditioned'")
    
    # Get global directory paths
    global results_dir, plots_dir, logs_dir
    
    # Set up the center
    if center is None:
        center = vector(RDF, [0] * dim)
    else:
        # If center is a list, convert to SageMath vector
        if isinstance(center, list):
            if len(center) != dim:
                raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({dim})")
            center = vector(RDF, center)
        elif isinstance(center, Vector):
            # Ensure correct field type
            center = vector(RDF, center)
        else:
            raise ValueError(f"Center must be None, list, or SageMath vector, got {type(center)}")
    
    # Create the lattice basis
    lattice_basis = create_lattice_basis(dim, basis_type)
    
    # Experiment name for file outputs
    experiment_name = f"{basis_type}_{dim}d_{float(sigma):.3g}"
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Parameters: dim={dim}, sigma={sigma}, samples={num_samples}, basis={basis_type}")
    logger.info(f"Center: {center}")
    
    # Run IMHK sampler
    logger.info("Running IMHK sampler...")
    imhk_start = time.time()
    
    # Import the new wrapper function
    from samplers import imhk_sampler_wrapper
    
    # Use the wrapper that handles different basis types
    imhk_samples, imhk_metadata = imhk_sampler_wrapper(
        basis_info=lattice_basis,
        sigma=sigma,
        num_samples=num_samples,
        center=center,
        burn_in=1000,  # Standard burn-in
        basis_type=basis_type
    )
    
    imhk_time = time.time() - imhk_start
    
    # Convert samples to explicit vectors if necessary
    if hasattr(imhk_samples, 'rows'):
        imhk_samples = [vector(row) for row in imhk_samples.rows()]
    
    logger.info(f"IMHK sampling completed in {imhk_time:.2f} seconds")
    logger.info(f"IMHK acceptance rate: {imhk_metadata.get('acceptance_rate', 0):.2%}")
    
    # Initialize results dictionary
    results = {
        'experiment_name': experiment_name,
        'dimension': dim,
        'sigma': float(sigma),
        'basis_type': basis_type,
        'num_samples': num_samples,
        'center': [float(x) for x in center],
        'imhk_time': imhk_time,
        'imhk_acceptance_rate': imhk_metadata.get('acceptance_rate', 0),
        'imhk_ess': {},
        'imhk_autocorr': {},
        'imhk_tv_distance': None,
        'imhk_kl_divergence': None
    }
    
    # Run Klein sampler for comparison if requested
    klein_samples = None
    if compare_with_klein:
        try:
            logger.info("Running Klein sampler for comparison...")
            klein_start = time.time()
            
            # Use the Klein wrapper that handles different basis types
            klein_samples_array, klein_metadata = klein_sampler_wrapper(
                basis_info=lattice_basis,
                sigma=sigma,
                num_samples=num_samples,
                center=center,
                basis_type=basis_type
            )
            
            # Convert to list of vectors for compatibility
            klein_samples = [vector(row) for row in klein_samples_array]
            
            klein_time = time.time() - klein_start
            
            logger.info(f"Klein sampling completed in {klein_time:.2f} seconds")
            
            results['klein_time'] = klein_time
            results['klein_tv_distance'] = None
            results['klein_kl_divergence'] = None
            results['speedup_ratio'] = klein_time / imhk_time if imhk_time > 0 else float('inf')
            
        except Exception as e:
            logger.warning(f"Klein sampler failed: {e}")
            klein_samples = None
    
    # Compute statistical metrics for IMHK samples
    logger.info("Computing statistical metrics...")
    
    # Calculate TV distance and KL divergence
    try:
        tv_distance = compute_total_variation_distance(
            samples=imhk_samples[:min(1000, len(imhk_samples))],  # Limit for efficiency
            sigma=sigma,
            lattice_basis=lattice_basis,
            center=center
        )
        results['imhk_tv_distance'] = float(tv_distance) if tv_distance is not None else None
        logger.info(f"IMHK TV distance: {tv_distance:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute TV distance for IMHK: {e}")
        results['imhk_tv_distance'] = None
    
    try:
        kl_divergence = compute_kl_divergence(
            samples=imhk_samples[:min(1000, len(imhk_samples))],
            sigma=sigma,
            lattice_basis=lattice_basis,
            center=center
        )
        results['imhk_kl_divergence'] = float(kl_divergence) if kl_divergence is not None else None
        logger.info(f"IMHK KL divergence: {kl_divergence:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute KL divergence for IMHK: {e}")
        results['imhk_kl_divergence'] = None
    
    # Compute metrics for Klein samples if available
    if klein_samples is not None and len(klein_samples) > 0:
        try:
            klein_tv = compute_total_variation_distance(
                samples=klein_samples[:min(1000, len(klein_samples))],
                sigma=sigma,
                lattice_basis=lattice_basis,
                center=center
            )
            results['klein_tv_distance'] = float(klein_tv) if klein_tv is not None else None
            logger.info(f"Klein TV distance: {klein_tv:.6f}")
        except Exception as e:
            logger.error(f"Failed to compute TV distance for Klein: {e}")
            results['klein_tv_distance'] = None
        
        try:
            klein_kl = compute_kl_divergence(
                samples=klein_samples[:min(1000, len(klein_samples))],
                sigma=sigma,
                lattice_basis=lattice_basis,
                center=center
            )
            results['klein_kl_divergence'] = float(klein_kl) if klein_kl is not None else None
            logger.info(f"Klein KL divergence: {klein_kl:.6f}")
        except Exception as e:
            logger.error(f"Failed to compute KL divergence for Klein: {e}")
            results['klein_kl_divergence'] = None
    
    # Compute autocorrelation and ESS for each dimension
    if imhk_metadata.get('trace') is not None:
        try:
            trace_data = np.array(imhk_metadata['trace'])
            for i in range(min(dim, trace_data.shape[1])):
                component_trace = trace_data[:, i]
                autocorr = compute_autocorrelation(component_trace.reshape(-1, 1))
                ess = compute_ess(component_trace.reshape(-1, 1))
                results['imhk_autocorr'][f'dim_{i}'] = autocorr[0][:10].tolist()  # First 10 lags
                results['imhk_ess'][f'dim_{i}'] = float(ess[0])
            
            # Compute average ESS
            if results['imhk_ess']:
                results['imhk_average_ess'] = float(np.mean(list(results['imhk_ess'].values())))
        except Exception as e:
            logger.error(f"Failed to compute ESS/autocorrelation: {e}")
    
    # Generate diagnostic plots if requested
    if plot_results and plots_dir is not None and imhk_metadata.get('trace') is not None:
        logger.info("Generating diagnostic plots...")
        try:
            # Plot trace (first few dimensions)
            plot_trace(imhk_metadata['trace'], os.path.join(plots_dir, f"trace_{experiment_name}.png"))
            
            # Plot autocorrelation
            plot_autocorrelation(imhk_metadata['trace'], os.path.join(plots_dir, f"autocorr_{experiment_name}.png"))
            
            # Plot acceptance rate over time
            if imhk_metadata.get('acceptance_trace') is not None:
                plot_acceptance_trace(
                    imhk_metadata['acceptance_trace'],
                    os.path.join(plots_dir, f"acceptance_{experiment_name}.png")
                )
            
            # Plot samples visualization based on dimension
            if dim == 2:
                plot_2d_samples(imhk_samples, sigma, os.path.join(plots_dir, f"samples_imhk_{experiment_name}.png"))
                if klein_samples is not None:
                    plot_2d_samples(klein_samples, sigma, os.path.join(plots_dir, f"samples_klein_{experiment_name}.png"))
            
            elif dim == 3:
                plot_3d_samples(imhk_samples, sigma, os.path.join(plots_dir, f"samples_imhk_{experiment_name}.png"))
                if klein_samples is not None:
                    plot_3d_samples(klein_samples, sigma, os.path.join(plots_dir, f"samples_klein_{experiment_name}.png"))
            
            else:
                # For higher dimensions, plot 2D projections
                plot_2d_projections(imhk_samples, sigma, os.path.join(plots_dir, f"proj_imhk_{experiment_name}.png"))
                if klein_samples is not None:
                    plot_2d_projections(klein_samples, sigma, os.path.join(plots_dir, f"proj_klein_{experiment_name}.png"))
                
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    # Save detailed results
    if save_results:
        # Save as JSON
        try:
            json_path = os.path.join(data_dir, f"{experiment_name}_results.json")
            # Create a JSON-serializable version
            json_results = {k: v for k, v in results.items() if v is not None}
            # Convert numpy arrays to lists for JSON serialization
            for key in json_results:
                if isinstance(json_results[key], dict):
                    for subkey in json_results[key]:
                        if isinstance(json_results[key][subkey], np.ndarray):
                            json_results[key][subkey] = json_results[key][subkey].tolist()
                elif isinstance(json_results[key], np.ndarray):
                    json_results[key] = json_results[key].tolist()
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved JSON results to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON results: {e}")
        
        # Save as pickle (includes sample data)
        try:
            pickle_path = os.path.join(data_dir, f"{experiment_name}_full.pkl")
            full_results = {
                'config': results,
                'imhk_samples': imhk_samples,
                'imhk_metadata': imhk_metadata,
                'klein_samples': klein_samples,
                'lattice_basis': lattice_basis
            }
            with open(pickle_path, 'wb') as f:
                pickle.dump(full_results, f)
            logger.info(f"Saved full results to {pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save pickle results: {e}")
    
    return results

# The rest of the file follows with the same modifications...
# (removing sklearn imports and PCA-related functions)

# Copy calculate_smoothing_parameter function from original
def calculate_smoothing_parameter(lattice_basis, epsilon=0.01):
    """
    Calculate the smoothing parameter η_ε(Λ) for a given lattice.
    
    Cryptographic Relevance:
    The smoothing parameter is fundamental in lattice-based cryptography as it:
    - Determines the minimum Gaussian width needed for uniform-like behavior
    - Appears in security reductions for LWE and SIS problems
    - Guides parameter selection in schemes like BLISS and FALCON
    - Ensures statistical closeness to continuous Gaussians
    
    Mathematical Definition:
    For a lattice Λ and ε > 0, the smoothing parameter η_ε(Λ) is the smallest s > 0
    such that ρ_{1/s}(Λ* \ {0}) ≤ ε, where Λ* is the dual lattice.
    
    Args:
        lattice_basis: The basis matrix of the lattice Λ
        epsilon: The smoothing parameter precision (default: 0.01)
                Smaller values give tighter bounds but require larger parameters
        
    Returns:
        The smoothing parameter η_ε(Λ) as a float
        
    Security Considerations:
    - For cryptographic security, typically ε ∈ [2^-40, 2^-100]
    - The Gaussian parameter σ should satisfy σ ≥ η_ε(Λ) for security
    - Common practice: σ = α·η_ε(Λ) where α > 1 (often α ∈ [1.5, 3])
    """
    from sage.all import matrix, QQ, sqrt, pi, Matrix
    
    # Validate inputs
    try:
        # Convert to matrix if needed
        if not hasattr(lattice_basis, 'nrows'):
            lattice_basis = matrix(lattice_basis)
    except Exception as e:
        raise ValueError(f"lattice_basis must be a matrix or convertible to one: {e}")
    
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
    
    n = lattice_basis.nrows()
    
    # For non-square matrices, check if full rank
    if lattice_basis.ncols() != n:
        raise ValueError("Lattice basis must be square")
    
    if lattice_basis.rank() < n:
        raise ValueError("Lattice basis must be full rank")
    
    # Use GSO for better numerical stability with general lattices
    try:
        logger.info(f"Computing smoothing parameter for {n}×{n} lattice with ε={epsilon}")
        
        # Compute GSO
        B_gso, mu = lattice_basis.gram_schmidt()
        
        # Find minimum GSO vector length
        min_gso_length = min(vector(b).norm() for b in B_gso)
        
        # Upper bound based on GSO
        from sage.all import log, sqrt, pi
        eta_upper = sqrt(log(2*n * (1 + 1/epsilon)) / pi) / min_gso_length
        
        logger.info(f"GSO-based upper bound: η_ε(Λ) ≤ {eta_upper}")
        
        # For special lattices, compute tighter bounds
        if lattice_basis.is_sparse() or (lattice_basis * lattice_basis.transpose()).is_diagonal():
            # For diagonal or nearly diagonal lattices
            eigenvalues = (lattice_basis * lattice_basis.transpose()).eigenvalues()
            min_eigenvalue = min(abs(ev) for ev in eigenvalues if abs(ev) > 1e-10)
            eta_tight = sqrt(log(2*n * (1 + 1/epsilon)) / pi) / sqrt(min_eigenvalue)
            logger.info(f"Tight bound for special lattice: η_ε(Λ) ≈ {eta_tight}")
            return float(eta_tight)
        
        # For general lattices, use the GSO bound with a safety factor
        safety_factor = 1.2  # Add 20% margin for numerical stability
        eta_final = float(eta_upper * safety_factor)
        
        logger.info(f"Final smoothing parameter: η_ε(Λ) = {eta_final}")
        return eta_final
        
    except Exception as e:
        logger.error(f"Error computing smoothing parameter: {e}")
        raise RuntimeError(f"Failed to compute smoothing parameter: {e}")
        
# Add the compare_tv_distance_vs_sigma function from the original
def compare_tv_distance_vs_sigma(
    dimensions=[8, 16, 32, 64], 
    basis_types=['identity', 'skewed', 'ill-conditioned', 'q-ary'],
    sigma_eta_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    num_samples=2000,
    plot_results=True,
    output_dir=None
):
    """
    Compare total variation distance between IMHK and Klein samplers across different sigma values.
    
    This function analyzes how the total variation distance varies with the ratio σ/η
    for different dimensions and lattice types, providing insights into:
    - Optimal parameter ranges for cryptographic applications
    - Trade-offs between security and efficiency
    - Performance differences between samplers
    
    Cryptographic Relevance:
    - Lower TV distance indicates better approximation to ideal discrete Gaussian
    - Critical for security proofs in lattice-based schemes
    - Helps determine minimal secure parameters
    
    Args:
        dimensions: List of lattice dimensions to test
        basis_types: List of basis types to evaluate
        sigma_eta_ratios: List of σ/η ratios to test
        num_samples: Number of samples per experiment
        plot_results: Whether to generate plots
        output_dir: Directory for saving results
        
    Returns:
        Dictionary mapping (dim, basis_type, ratio) to experiment results
    """
    logger.info("Starting TV distance comparison across sigma values")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Basis types: {basis_types}")
    logger.info(f"Sigma/eta ratios: {sigma_eta_ratios}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Run experiments
    for dim in dimensions:
        logger.info(f"Processing dimension {dim}")
        
        # Calculate baseline smoothing parameter for identity lattice
        identity_basis = create_lattice_basis(dim, 'identity')
        base_eta = calculate_smoothing_parameter(identity_basis, epsilon=1/16)
        logger.info(f"Smoothing parameter η_6.25e-02(Λ) = {float(base_eta):.4f} for dimension {dim}")
        
        for basis_type in basis_types:
            logger.info(f"  Processing basis type: {basis_type}")
            
            # Create lattice basis
            try:
                lattice_basis = create_lattice_basis(dim, basis_type)
            except Exception as e:
                logger.error(f"Failed to create {basis_type} basis for dimension {dim}: {e}")
                continue
            
            # Calculate actual smoothing parameter for this basis
            try:
                eta = calculate_smoothing_parameter(lattice_basis, epsilon=1/16)
            except Exception as e:
                logger.warning(f"Failed to calculate smoothing parameter for {basis_type} basis, using base value")
                eta = base_eta
            
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                logger.info(f"    Testing sigma = {float(sigma):.4f} (ratio = {ratio})")
                
                key = (dim, basis_type, ratio)
                
                try:
                    # Run experiment
                    exp_results = run_experiment(
                        dim=dim,
                        sigma=sigma,
                        num_samples=num_samples,
                        basis_type=basis_type,
                        compare_with_klein=True,
                        plot_results=False  # We'll do custom plots
                    )
                    
                    # Store results
                    results[key] = {
                        'dimension': dim,
                        'basis_type': basis_type,
                        'sigma_eta_ratio': ratio,
                        'sigma': float(sigma),
                        'eta': float(eta),
                        'imhk_tv_distance': exp_results.get('imhk_tv_distance'),
                        'klein_tv_distance': exp_results.get('klein_tv_distance'),
                        'imhk_acceptance_rate': exp_results.get('imhk_acceptance_rate'),
                        'time_ratio': exp_results.get('speedup_ratio')
                    }
                    
                except Exception as e:
                    logger.error(f"    Error running experiment: {e}")
                    results[key] = {
                        'dimension': dim,
                        'basis_type': basis_type,
                        'sigma_eta_ratio': ratio,
                        'error': str(e)
                    }
    
    # Save results
    try:
        # Save as JSON (convert to serializable format)
        json_results = {}
        for key, value in results.items():
            json_key = f"{key[0]}_{key[1]}_{key[2]}"
            json_results[json_key] = value
        
        json_path = os.path.join(output_dir, "tv_distance_comparison.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved JSON results to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
    
    # Save as pickle (preserves full structure)
    pickle_path = os.path.join(output_dir, "tv_distance_comparison.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved detailed results to {pickle_path}")
    
    # Generate plots if requested
    if plot_results:
        _plot_tv_distance_comparison(results, output_dir)
    
    return results

def _plot_tv_distance_comparison(results, output_dir):
    """Generate comparison plots for TV distance analysis."""
    logger.info("Generating TV distance comparison plots")
    
    # Ensure plots directory exists
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group results by dimension and basis type
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    
    for (dim, basis_type, ratio), result in results.items():
        if 'error' not in result:
            grouped[dim][basis_type].append((ratio, result))
    
    # Plot TV distance vs sigma/eta ratio for each dimension
    for dim in sorted(grouped.keys()):
        plt.figure(figsize=(10, 6))
        
        for basis_type in sorted(grouped[dim].keys()):
            data = sorted(grouped[dim][basis_type], key=lambda x: x[0])
            ratios = [d[0] for d in data]
            imhk_tv = [d[1]['imhk_tv_distance'] for d in data if d[1]['imhk_tv_distance'] is not None]
            klein_tv = [d[1]['klein_tv_distance'] for d in data if d[1]['klein_tv_distance'] is not None]
            
            if imhk_tv:
                plt.plot(ratios[:len(imhk_tv)], imhk_tv, 'o-', label=f'IMHK ({basis_type})')
            if klein_tv:
                plt.plot(ratios[:len(klein_tv)], klein_tv, 's--', label=f'Klein ({basis_type})')
        
        plt.xlabel('σ/η Ratio')
        plt.ylabel('Total Variation Distance')
        plt.title(f'TV Distance vs σ/η Ratio (Dimension {dim})')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'tv_distance_dim{dim}.png'), dpi=150)
        plt.close()
    
    # Plot acceptance rate vs sigma/eta ratio
    for dim in sorted(grouped.keys()):
        plt.figure(figsize=(10, 6))
        
        for basis_type in sorted(grouped[dim].keys()):
            data = sorted(grouped[dim][basis_type], key=lambda x: x[0])
            ratios = [d[0] for d in data]
            acceptance_rates = [d[1]['imhk_acceptance_rate'] for d in data 
                              if d[1].get('imhk_acceptance_rate') is not None]
            
            if acceptance_rates:
                plt.plot(ratios[:len(acceptance_rates)], acceptance_rates, 'o-', 
                        label=f'{basis_type}')
        
        plt.xlabel('σ/η Ratio')
        plt.ylabel('Acceptance Rate')
        plt.title(f'IMHK Acceptance Rate vs σ/η Ratio (Dimension {dim})')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'acceptance_rate_dim{dim}.png'), dpi=150)
        plt.close()
    
    # Create summary plot comparing all dimensions
    plt.figure(figsize=(12, 8))
    
    for basis_type in ['identity', 'skewed', 'ill-conditioned']:
        for dim in sorted(grouped.keys()):
            if basis_type in grouped[dim]:
                data = sorted(grouped[dim][basis_type], key=lambda x: x[0])
                ratios = [d[0] for d in data]
                imhk_tv = [d[1]['imhk_tv_distance'] for d in data 
                          if d[1]['imhk_tv_distance'] is not None]
                
                if imhk_tv:
                    plt.plot(ratios[:len(imhk_tv)], imhk_tv, 'o-', 
                            label=f'Dim {dim} ({basis_type})')
    
    plt.xlabel('σ/η Ratio')
    plt.ylabel('Total Variation Distance')
    plt.title('TV Distance Comparison Across Dimensions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tv_distance_summary.png'), dpi=150)
    plt.close()
    
    logger.info(f"Plots saved to {plots_dir}")

# Main execution
if __name__ == "__main__":
    # Example usage
    logger.info("Running IMHK sampler experiments")
    
    # Initialize directories
    init_directories("results/main_experiments")
    
    # Run a simple experiment
    result = run_experiment(
        dim=8,
        sigma=2.0,
        num_samples=1000,
        basis_type='identity'
    )
    
    print(f"Experiment completed: {result['experiment_name']}")
    print(f"IMHK acceptance rate: {result['imhk_acceptance_rate']:.2%}")
    print(f"TV distance: {result.get('imhk_tv_distance', 'N/A')}")