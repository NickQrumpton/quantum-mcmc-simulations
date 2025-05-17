import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sage.all import *
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import multiprocessing as mp
from functools import partial
import os
import logging

# Create necessary directories
base_dir = Path(__file__).resolve().parent.parent
results_dir = base_dir / "results"
logs_dir = results_dir / "logs"
plots_dir = results_dir / "plots"

for directory in [results_dir, logs_dir, plots_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging with proper path resolution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lattice_experiments")


def validate_basis_type(basis_type: str) -> bool:
    """
    Validate that the basis type is supported.
    
    Args:
        basis_type: String specifying the lattice basis type
        
    Returns:
        True if valid, False otherwise
    """
    return basis_type in ['identity', 'skewed', 'ill-conditioned']


def create_lattice_basis(dim: int, basis_type: str) -> Matrix:
    """
    Create a lattice basis matrix of the specified type.
    
    Args:
        dim: Dimension of the lattice (must be ≥ 2)
        basis_type: Type of lattice basis to create
        
    Returns:
        The lattice basis matrix
        
    Raises:
        ValueError: If basis_type is not recognized or dim < 2
    """
    if not isinstance(dim, (int, Integer)) or dim < 2:
        raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    
    if not validate_basis_type(basis_type):
        raise ValueError(f"Unknown basis type: {basis_type}. Valid options are 'identity', 'skewed', 'ill-conditioned'")
    
    # Create the lattice basis according to type
    if basis_type == 'identity':
        # Standard orthogonal basis - ideal case
        B = matrix.identity(RR, dim)
    elif basis_type == 'skewed':
        # Basis with non-orthogonal vectors - common in cryptographic applications
        B = matrix.identity(RR, dim)
        B[0, 1] = 1.5  # Add some skew to simulate non-orthogonality
        if dim >= 3:
            B[0, 2] = 0.5  # Additional skew in higher dimensions
    elif basis_type == 'ill-conditioned':
        # Basis with poor conditioning - challenging for sampling algorithms
        # Models cases where some lattice vectors are much longer than others
        B = matrix.identity(RR, dim)
        B[0, 0] = 10.0  # First vector is much longer
        B[1, 1] = 0.1   # Second vector is much shorter
    
    return B


def calculate_smoothing_parameter(dim: int, epsilon: float = 0.01) -> RealNumber:
    """
    Calculate the smoothing parameter for a lattice.
    
    Mathematical Context:
    The smoothing parameter η_ε(Λ) is a fundamental concept in lattice-based cryptography.
    It represents the Gaussian parameter at which the discrete Gaussian distribution over
    the dual lattice Λ* appears nearly uniform modulo the lattice.
    
    For an identity basis, the smoothing parameter is approximated as:
    η_ε(Λ) ≈ sqrt(ln(2n/ε)/π)
    
    Where:
    - n is the lattice dimension
    - ε is a small constant, typically 2^(-n)
    
    Cryptographic Relevance:
    In lattice-based cryptography, sampling with σ > η_ε(Λ) ensures statistical
    properties required for security proofs in schemes like:
    - Ring-LWE encryption
    - Lattice-based signature schemes
    - Trapdoor functions
    
    Args:
        dim: Dimension of the lattice
        epsilon: Small constant (default: 0.01)
        
    Returns:
        The approximate smoothing parameter
    """
    return sqrt(log(2*dim/epsilon)/pi)


def run_experiment(
    dim: int, 
    sigma: float, 
    num_samples: int, 
    basis_type: str = 'identity', 
    compare_with_klein: bool = True, 
    center: Optional[Union[Vector, List[float]]] = None
) -> Dict[str, Any]:
    """
    Run a complete experiment with IMHK sampling and analysis.
    
    Mathematical Context:
    This function evaluates the quality of discrete Gaussian sampling over lattices
    by comparing the Independent Metropolis-Hastings-Klein (IMHK) algorithm with
    Klein's algorithm. The quality is assessed using:
    - Total Variation distance to the ideal distribution
    - KL divergence to measure information-theoretic similarity
    - Effective Sample Size (ESS) to quantify the independence of samples
    
    Cryptographic Relevance:
    Discrete Gaussian sampling over lattices is fundamental to lattice-based cryptography:
    - Used in signature schemes (e.g., FALCON, CRYSTALS-Dilithium)
    - Essential for key generation in encryption schemes
    - Critical for security proofs and concrete security parameters
    - Affects resistance to side-channel attacks
    
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
    # Import required functions with proper module paths
    from imhk_sampler.samplers import imhk_sampler, klein_sampler
    from imhk_sampler.diagnostics import plot_trace, plot_autocorrelation, plot_acceptance_trace, compute_autocorrelation, compute_ess
    from imhk_sampler.visualization import plot_2d_samples, plot_3d_samples, plot_2d_projections, plot_pca_projection
    from imhk_sampler.stats import compute_total_variation_distance, compute_kl_divergence
    
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
        center = vector(RR, [0] * dim)
    else:
        # If center is a list, convert to SageMath vector
        if isinstance(center, list):
            if len(center) != dim:
                raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({dim})")
            center = vector(RR, center)
        elif isinstance(center, Vector):
            if len(center) != dim:
                raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({dim})")
        else:
            raise TypeError(f"Center must be a list or SageMath vector, got {type(center)}")
    
    # Create the lattice basis
    B = create_lattice_basis(dim, basis_type)
    
    # Create a unique experiment name
    experiment_name = f"dim{dim}_sigma{sigma}_{basis_type}"
    if any(c != 0 for c in center):
        experiment_name += f"_center{'_'.join(str(float(c)) for c in center)}"
    
    # Calculate smoothing parameter for reference
    epsilon = 0.01  # A small constant
    smoothing_param = calculate_smoothing_parameter(dim, epsilon)
    
    logger.info(f"Running experiment: dim={dim}, sigma={sigma}, basis={basis_type}")
    logger.info(f"Smoothing parameter η_{epsilon}(Λ) ≈ {smoothing_param:.4f} for reference")
    logger.info(f"σ/η ratio: {sigma/smoothing_param:.4f}")
    
    # Run IMHK sampler
    burn_in = min(5000, num_samples)  # Use appropriate burn-in
    start_time = time.time()
    try:
        imhk_samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in)
        imhk_time = time.time() - start_time
        logger.info(f"IMHK sampling completed in {imhk_time:.2f} seconds, acceptance rate: {acceptance_rate:.4f}")
    except Exception as e:
        logger.error(f"IMHK sampling failed: {str(e)}")
        raise
    
    # Convert sample lists to NumPy arrays for more efficient processing
    imhk_samples_np = np.array([[float(x_i) for x_i in x] for x in imhk_samples])
    
    # Run Klein sampler for comparison if requested
    klein_samples = None
    klein_time = None
    if compare_with_klein:
        start_time = time.time()
        try:
            # Use list comprehension for Klein sampling
            klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
            klein_time = time.time() - start_time
            logger.info(f"Klein sampling completed in {klein_time:.2f} seconds")
            
            # Convert to NumPy for efficiency
            klein_samples_np = np.array([[float(x_i) for x_i in x] for x in klein_samples])
        except Exception as e:
            logger.error(f"Klein sampling failed: {str(e)}")
            logger.warning("Continuing without Klein comparison")
            compare_with_klein = False
    
    # Run diagnostics and create visualizations
    try:
        # Analyze acceptance rate over time
        plot_acceptance_trace(all_accepts, f"acceptance_trace_{experiment_name}.png")
        
        # Trace plots
        plot_trace(imhk_samples, f"trace_imhk_{experiment_name}.png", 
                f"IMHK Sample Trace (σ={sigma}, {basis_type} basis)")
        
        # Autocorrelation
        acf_by_dim = compute_autocorrelation(imhk_samples)
        plot_autocorrelation(acf_by_dim, f"acf_imhk_{experiment_name}.png", 
                          f"IMHK Autocorrelation (σ={sigma}, {basis_type} basis)")
        
        # Effective Sample Size
        ess_values = compute_ess(imhk_samples)
        logger.info(f"IMHK Effective Sample Size: {ess_values}")
        
        # Visualization based on dimension
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
        
        logger.info("Visualizations completed successfully")
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        logger.warning("Continuing with statistical analysis")
    
    # Compute statistical distances
    try:
        tv_distance = compute_total_variation_distance(imhk_samples, sigma, B, center)
        logger.info(f"IMHK Total Variation distance: {tv_distance:.6f}")
        
        # Compute KL divergence for small dimensions
        kl_divergence = None
        if dim <= 3:  # Only compute for small dimensions due to computational complexity
            kl_divergence = compute_kl_divergence(imhk_samples, sigma, B, center)
            logger.info(f"IMHK KL divergence: {kl_divergence:.6f}")
    except Exception as e:
        logger.error(f"Error computing IMHK statistical distances: {str(e)}")
        tv_distance = None
        kl_divergence = None
    
    # Compile results
    results = {
        'dimension': dim,
        'sigma': float(sigma),
        'basis_type': basis_type,
        'center': [float(c) for c in center],
        'smoothing_parameter': float(smoothing_param),
        'sigma_smoothing_ratio': float(sigma/smoothing_param),
        'num_samples': num_samples,
        'burn_in': burn_in,
        'imhk_acceptance_rate': float(acceptance_rate),
        'imhk_time': float(imhk_time),
        'imhk_ess': [float(ess) for ess in ess_values],
        'imhk_tv_distance': float(tv_distance) if tv_distance is not None else None,
        'imhk_kl_divergence': float(kl_divergence) if kl_divergence is not None else None
    }
    
    # Klein comparison results
    if compare_with_klein and klein_samples is not None:
        try:
            # Compute TV distance for Klein samples
            klein_tv_distance = compute_total_variation_distance(klein_samples, sigma, B, center)
            results['klein_time'] = float(klein_time)
            results['klein_tv_distance'] = float(klein_tv_distance)
            logger.info(f"Klein Total Variation distance: {klein_tv_distance:.6f}")
            
            # Compute KL divergence for Klein samples if feasible
            if dim <= 3:
                klein_kl_divergence = compute_kl_divergence(klein_samples, sigma, B, center)
                results['klein_kl_divergence'] = float(klein_kl_divergence)
                logger.info(f"Klein KL divergence: {klein_kl_divergence:.6f}")
        except Exception as e:
            logger.error(f"Error computing Klein statistical distances: {str(e)}")
    
    # Save results to log file
    try:
        with open(logs_dir / f"experiment_{experiment_name}.txt", "w") as f:
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
            
            if tv_distance is not None:
                f.write(f"Total Variation distance: {tv_distance:.6f}\n")
            
            if kl_divergence is not None:
                f.write(f"KL divergence: {kl_divergence:.6f}\n")
            
            if compare_with_klein and klein_samples is not None:
                f.write("\n=== Klein Sampler Results ===\n")
                f.write(f"Sampling time: {klein_time:.6f} seconds\n")
                
                if 'klein_tv_distance' in results:
                    f.write(f"Total Variation distance: {results['klein_tv_distance']:.6f}\n")
                
                if 'klein_kl_divergence' in results:
                    f.write(f"KL divergence: {results['klein_kl_divergence']:.6f}\n")
                
                f.write("\n=== Comparison ===\n")
                if klein_time > 0:
                    f.write(f"IMHK/Klein time ratio: {imhk_time/klein_time:.6f}\n")
                
                if 'klein_tv_distance' in results and results['klein_tv_distance'] > 0:
                    f.write(f"IMHK/Klein TV distance ratio: {tv_distance/results['klein_tv_distance']:.6f}\n")
                
                if 'klein_kl_divergence' in results and results['klein_kl_divergence'] > 0:
                    f.write(f"IMHK/Klein KL divergence ratio: {kl_divergence/results['klein_kl_divergence']:.6f}\n")
    except Exception as e:
        logger.error(f"Error writing log file: {str(e)}")
    
    # Save all data for later analysis
    try:
        with open(logs_dir / f"experiment_{experiment_name}.pickle", "wb") as f:
            pickle.dump(results, f)
    except Exception as e:
        logger.error(f"Error saving pickle file: {str(e)}")
    
    return results


def _run_experiment_wrapper(params: Tuple[int, float, str, Vector, int]) -> Dict[str, Any]:
    """
    Wrapper function for running experiments in parallel.
    
    Args:
        params: Tuple containing (dimension, sigma, basis_type, center, num_samples)
        
    Returns:
        Experiment results
    """
    dim, sigma, basis_type, center, num_samples = params
    try:
        return run_experiment(
            dim=dim, 
            sigma=sigma, 
            num_samples=num_samples,
            basis_type=basis_type,
            compare_with_klein=True,
            center=center
        )
    except Exception as e:
        logger.error(f"Error in experiment: dim={dim}, sigma={sigma}, basis={basis_type}: {str(e)}")
        return {
            'dimension': dim,
            'sigma': sigma,
            'basis_type': basis_type,
            'center': center,
            'error': str(e)
        }


def parameter_sweep(
    dimensions: Optional[List[int]] = None, 
    sigmas: Optional[List[float]] = None, 
    basis_types: Optional[List[str]] = None, 
    centers: Optional[Dict[int, List[Vector]]] = None, 
    num_samples: int = 1000,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[Tuple, Dict[str, Any]]:
    """
    Perform a parameter sweep across different dimensions, sigmas, basis types, and centers.
    
    Mathematical Context:
    This function systematically explores the parameter space to evaluate:
    - How sampling quality varies with the Gaussian parameter σ
    - The impact of lattice dimension on sampling efficiency and accuracy
    - How different basis types (well-conditioned vs. ill-conditioned) affect sampling
    - The effect of off-center distributions on sampling algorithms
    
    Cryptographic Relevance:
    Parameter sweeps are essential for:
    - Determining optimal parameters for lattice-based cryptographic schemes
    - Evaluating the trade-off between security (higher σ) and efficiency
    - Understanding how basis quality affects real-world implementations
    - Establishing confidence in security parameters across different scenarios
    
    Args:
        dimensions: List of dimensions to test (default: [2, 3, 4])
        sigmas: List of sigma values to test (default: [0.5, 1.0, 2.0, 5.0])
        basis_types: List of basis types to test (default: ['identity', 'skewed', 'ill-conditioned'])
        centers: Dictionary mapping dimensions to lists of centers (default: origin for each dimension)
        num_samples: Number of samples to generate for each configuration (default: 1000)
        parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: CPU count-1)
        
    Returns:
        A dictionary of results indexed by configuration
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if dimensions is None:
        dimensions = [2, 3, 4]
    else:
        for dim in dimensions:
            if not isinstance(dim, (int, Integer)) or dim < 2:
                raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0, 5.0]
    else:
        for sigma in sigmas:
            if not isinstance(sigma, (float, RealNumber)) or sigma <= 0:
                raise ValueError(f"Sigma must be positive, got {sigma}")
    
    if basis_types is None:
        basis_types = ['identity', 'skewed', 'ill-conditioned']
    else:
        for basis_type in basis_types:
            if not validate_basis_type(basis_type):
                raise ValueError(f"Unknown basis type: {basis_type}")
    
    if centers is None:
        centers = {dim: [vector(RR, [0] * dim)] for dim in dimensions}
    elif isinstance(centers, list):
        # If centers is a list of vectors, assume it applies to all dimensions
        centers = {dim: [vector(RR, c) for c in centers if len(c) == dim] for dim in dimensions}
    
    # Get global directory paths
    global logs_dir, plots_dir
    
    # Create experiment parameter combinations
    experiment_params = []
    for dim in dimensions:
        for sigma in sigmas:
            for basis_type in basis_types:
                for center in centers.get(dim, [vector(RR, [0] * dim)]):
                    experiment_params.append((dim, sigma, basis_type, center, num_samples))
    
    logger.info(f"Parameter sweep: {len(experiment_params)} configurations to test")
    
    results = {}
    
    # Create a summary file
    try:
        with open(logs_dir / "parameter_sweep_summary.txt", "w") as summary_file:
            summary_file.write("Parameter Sweep Summary\n")
            summary_file.write("=====================\n\n")
    except Exception as e:
        logger.error(f"Failed to create summary file: {e}")
        raise
    
    # Run experiments in parallel or sequentially
    if parallel and len(experiment_params) > 1:
        if max_workers is None:
            # Default to CPU count - 1 to avoid overwhelming the system
            max_workers = max(1, mp.cpu_count() - 1)
        
        logger.info(f"Using parallel processing with {max_workers} workers")
        
        try:
            with mp.Pool(processes=max_workers) as pool:
                experiment_results = pool.map(_run_experiment_wrapper, experiment_params)
            
            # Process results
            for params, result in zip(experiment_params, experiment_results):
                dim, sigma, basis_type, center, _ = params
                config_key = (dim, sigma, basis_type, tuple(center))
                results[config_key] = result
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing")
            
            # Fall back to sequential processing
            for params in experiment_params:
                dim, sigma, basis_type, center, _ = params
                config_key = (dim, sigma, basis_type, tuple(center))
                results[config_key] = _run_experiment_wrapper(params)
    else:
        # Sequential processing
        logger.info("Using sequential processing")
        for params in experiment_params:
            dim, sigma, basis_type, center, _ = params
            config_key = (dim, sigma, basis_type, tuple(center))
            results[config_key] = _run_experiment_wrapper(params)
    
    # Update summary with results
    try:
        with open(logs_dir / "parameter_sweep_summary.txt", "a") as summary_file:
            for config_key, result in results.items():
                dim, sigma, basis_type, center_tuple = config_key
                
                if 'error' in result:
                    summary_file.write(f"Configuration: dim={dim}, sigma={sigma}, ")
                    summary_file.write(f"basis={basis_type}, center={center_tuple}\n")
                    summary_file.write(f"ERROR: {result['error']}\n")
                    summary_file.write("---\n\n")
                    continue
                
                summary_file.write(f"Configuration: dim={dim}, sigma={sigma}, ")
                summary_file.write(f"basis={basis_type}, center={center_tuple}\n")
                summary_file.write(f"IMHK Acceptance Rate: {result.get('imhk_acceptance_rate', 'N/A')}\n")
                
                if 'imhk_tv_distance' in result and result['imhk_tv_distance'] is not None:
                    summary_file.write(f"IMHK Total Variation Distance: {result['imhk_tv_distance']:.6f}\n")
                
                if 'klein_tv_distance' in result and result['klein_tv_distance'] is not None:
                    summary_file.write(f"Klein Total Variation Distance: {result['klein_tv_distance']:.6f}\n")
                    
                    # Only compute ratio if both values are present and non-zero
                    if (result['imhk_tv_distance'] is not None and 
                        result['klein_tv_distance'] is not None and 
                        result['klein_tv_distance'] > 0):
                        ratio = result['imhk_tv_distance'] / result['klein_tv_distance']
                        summary_file.write(f"IMHK/Klein TV Ratio: {ratio:.4f}\n")
                
                summary_file.write("---\n\n")
    except Exception as e:
        logger.error(f"Error updating summary file: {e}")
    
    # Generate comparative plots
    try:
        plot_parameter_sweep_results(results, dimensions, sigmas, basis_types)
    except Exception as e:
        logger.error(f"Error generating parameter sweep plots: {e}")
    
    return results


def plot_parameter_sweep_results(
    results: Dict[Tuple, Dict[str, Any]], 
    dimensions: List[int], 
    sigmas: List[float], 
    basis_types: List[str]
) -> None:
    """
    Create comparative plots for the parameter sweep results.
    
    Cryptographic Relevance:
    These visualizations help researchers:
    - Identify optimal sampling parameters for lattice-based schemes
    - Understand the relationship between σ/η ratio and sampling quality
    - Visualize trade-offs between efficiency (acceptance rate) and accuracy (TV distance)
    - Compare IMHK and Klein samplers across different lattice configurations
    
    Args:
        results: Dictionary of results from parameter_sweep
        dimensions: List of dimensions tested
        sigmas: List of sigma values tested
        basis_types: List of basis types tested
        
    Returns:
        None (saves plots to files)
    """
    # Get global plot directory
    global plots_dir
    
    # Set plot style for publication quality
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'gray'
    })
    
    # Plot acceptance rate vs. sigma for each dimension and basis type
    for dim in dimensions:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            markers = ['o', 's', '^', 'D', 'x', '+']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, basis_type in enumerate(basis_types):
                # Extract data for this dimension and basis type
                x_data = []
                y_data = []
                
                for sigma in sigmas:
                    key = (dim, sigma, basis_type, tuple([0] * dim))
                    if key in results and 'imhk_acceptance_rate' in results[key]:
                        x_data.append(sigma)
                        y_data.append(results[key]['imhk_acceptance_rate'])
                
                if x_data:
                    marker = markers[i % len(markers)]
                    color = colors[i % len(colors)]
                    ax.plot(x_data, y_data, marker=marker, linestyle='-', color=color,
                           label=f"{basis_type}", linewidth=1.5, markersize=6)
            
            # Add smoothing parameter reference lines for each sigma/eta ratio
            epsilon = 0.01
            eta = calculate_smoothing_parameter(dim, epsilon)
            
            ratios = [1.0, 2.0, 4.0]
            for ratio in ratios:
                sigma_val = ratio * eta
                if min(sigmas) <= sigma_val <= max(sigmas):
                    ax.axvline(sigma_val, color='gray', linestyle='--', alpha=0.5, 
                             label=f"σ/η = {ratio}")
            
            ax.set_xlabel('Gaussian Parameter (σ)')
            ax.set_ylabel('IMHK Acceptance Rate')
            ax.set_title(f'Acceptance Rate vs. σ (Dimension {dim})')
            ax.grid(True, alpha=0.3)
            
            # Create a more compact legend
            ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'acceptance_vs_sigma_dim{dim}.png')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting acceptance rate for dimension {dim}: {e}")
    
    # Plot TV distance vs. sigma for each dimension and basis type
    for dim in dimensions:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            markers = ['o', 's', '^', 'D', 'x', '+']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, basis_type in enumerate(basis_types):
                # Extract data for this dimension and basis type
                x_data = []
                y_imhk = []
                y_klein = []
                
                for sigma in sigmas:
                    key = (dim, sigma, basis_type, tuple([0] * dim))
                    if key in results:
                        res = results[key]
                        if ('imhk_tv_distance' in res and res['imhk_tv_distance'] is not None and
                            'klein_tv_distance' in res and res['klein_tv_distance'] is not None):
                            x_data.append(sigma)
                            y_imhk.append(res['imhk_tv_distance'])
                            y_klein.append(res['klein_tv_distance'])
                
                if x_data:
                    marker_imhk = markers[i % len(markers)]
                    marker_klein = markers[(i + 1) % len(markers)]
                    color = colors[i % len(colors)]
                    
                    ax.plot(x_data, y_imhk, marker=marker_imhk, linestyle='-', color=color,
                           label=f"IMHK {basis_type}", linewidth=1.5, markersize=6)
                    ax.plot(x_data, y_klein, marker=marker_klein, linestyle='--', color=color,
                           label=f"Klein {basis_type}", linewidth=1.0, markersize=5, alpha=0.7)
            
            # Add smoothing parameter reference lines
            epsilon = 0.01
            eta = calculate_smoothing_parameter(dim, epsilon)
            
            ratios = [1.0, 2.0, 4.0]
            for ratio in ratios:
                sigma_val = ratio * eta
                if min(sigmas) <= sigma_val <= max(sigmas):
                    ax.axvline(sigma_val, color='gray', linestyle='--', alpha=0.5, 
                             label=f"σ/η = {ratio}")
            
            ax.set_xlabel('Gaussian Parameter (σ)')
            ax.set_ylabel('Total Variation Distance')
            ax.set_title(f'TV Distance vs. σ (Dimension {dim})')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis to log scale to better visualize small differences
            ax.set_yscale('log')
            
            # Create a more compact legend with two columns
            ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, ncol=2)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'tv_distance_vs_sigma_dim{dim}.png')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting TV distance for dimension {dim}: {e}")
    
    # Plot TV distance ratio (IMHK/Klein) vs. sigma
    for dim in dimensions:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            markers = ['o', 's', '^', 'D', 'x', '+']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, basis_type in enumerate(basis_types):
                # Extract data for this dimension and basis type
                x_data = []
                y_ratio = []
                
                for sigma in sigmas:
                    key = (dim, sigma, basis_type, tuple([0] * dim))
                    if key in results:
                        res = results[key]
                        if ('imhk_tv_distance' in res and res['imhk_tv_distance'] is not None and
                            'klein_tv_distance' in res and res['klein_tv_distance'] is not None and
                            res['klein_tv_distance'] > 0):
                            x_data.append(sigma)
                            ratio = res['imhk_tv_distance'] / res['klein_tv_distance']
                            y_ratio.append(ratio)
                
                if x_data:
                    marker = markers[i % len(markers)]
                    color = colors[i % len(colors)]
                    ax.plot(x_data, y_ratio, marker=marker, linestyle='-', color=color,
                           label=f"{basis_type}", linewidth=1.5, markersize=6)
            
            # Add smoothing parameter reference lines
            epsilon = 0.01
            eta = calculate_smoothing_parameter(dim, epsilon)
            
            ratios = [1.0, 2.0, 4.0]
            for ratio in ratios:
                sigma_val = ratio * eta
                if min(sigmas) <= sigma_val <= max(sigmas):
                    ax.axvline(sigma_val, color='gray', linestyle='--', alpha=0.5, 
                             label=f"σ/η = {ratio}")
            
            ax.set_xlabel('Gaussian Parameter (σ)')
            ax.set_ylabel('TV Distance Ratio (IMHK/Klein)')
            ax.set_title(f'Quality Improvement Ratio vs. σ (Dimension {dim})')
            ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Equal Quality')
            ax.grid(True, alpha=0.3)
            
            # Create a more compact legend
            ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'tv_ratio_vs_sigma_dim{dim}.png')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting TV ratio for dimension {dim}: {e}")