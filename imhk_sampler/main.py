import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from sage.all import *
from sage.matrix.matrix_space import MatrixSpace
from sage.modules.free_module_element import vector

from .samplers import imhk_sampler, klein_sampler
from .diagnostics import plot_trace, plot_autocorrelation, compute_autocorrelation
from .visualization import plot_2d_samples
from .stats import compute_total_variation_distance, compute_kl_divergence
from .experiments import parameter_sweep, compare_convergence_times, run_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/logs/main.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("imhk_main")


def run_basic_example() -> None:
    """
    Run a basic example of the IMHK sampler on a 2D lattice.
    
    Purpose:
    This function demonstrates the fundamental behavior of the Independent 
    Metropolis-Hastings-Klein (IMHK) sampler on a simple 2D lattice with an identity basis.
    It compares IMHK with Klein's algorithm in terms of sample quality and statistical
    properties.
    
    Relevance to Lattice-based Cryptography:
    High-quality discrete Gaussian sampling is essential for:
    - Lattice-based signature schemes (e.g., FALCON)
    - Security parameter selection in lattice-based encryption
    - Generating trapdoors with specific statistical properties
    - Understanding the practical performance of theoretical sampling algorithms
    
    This example helps visualize and quantify the sampling quality, which directly
    impacts the security and efficiency of lattice-based cryptographic primitives.
    
    Returns:
        None (generates plots and prints results)
    """
    logger.info("Running basic 2D IMHK example...")
    
    # Parameters
    dim = 2
    sigma = 2.0
    num_samples = 2000
    burn_in = 1000
    
    # Create output directories if they don't exist
    try:
        Path("results/plots").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}")
        raise
    
    # Identity basis (orthogonal basis vectors of unit length)
    B = matrix.identity(RR, dim)
    
    # Calculate smoothing parameter for reference
    # The smoothing parameter η_ε(Λ) ≈ sqrt(ln(2n/ε)/π) for identity basis
    epsilon = 0.01  # A small constant
    smoothing_param = sqrt(log(2*dim/epsilon)/pi)
    logger.info(f"Smoothing parameter: {smoothing_param:.4f}")
    logger.info(f"σ/η ratio: {sigma/smoothing_param:.4f}")
    
    try:
        # Run IMHK sampler
        samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
            B, sigma, num_samples, burn_in=burn_in)
        
        logger.info(f"Acceptance rate: {acceptance_rate:.4f}")
        
        # Plot trace to visualize the Markov chain behavior
        plot_trace(samples, "basic_example_trace.png", "IMHK Sample Trace (2D)")
        
        # Compute autocorrelation to assess sample independence
        acf_by_dim = compute_autocorrelation(samples)
        plot_autocorrelation(acf_by_dim, "basic_example_acf.png", "IMHK Autocorrelation (2D)")
        
        # Visualize the sample distribution
        plot_2d_samples(samples, sigma, "basic_example_samples.png", B, "IMHK Samples (2D)")
        
        # Run Klein sampler for comparison
        klein_samples = [klein_sampler(B, sigma) for _ in range(num_samples)]
        plot_2d_samples(klein_samples, sigma, "basic_example_klein.png", B, "Klein Samples (2D)")
        
        # Compute statistical distances to assess sampling quality
        tv_distance_imhk = compute_total_variation_distance(samples, sigma, B)
        tv_distance_klein = compute_total_variation_distance(klein_samples, sigma, B)
        
        logger.info(f"IMHK Total Variation distance: {tv_distance_imhk:.6f}")
        logger.info(f"Klein Total Variation distance: {tv_distance_klein:.6f}")
        
        # Compute KL divergence as another measure of distribution similarity
        kl_imhk = compute_kl_divergence(samples, sigma, B)
        kl_klein = compute_kl_divergence(klein_samples, sigma, B)
        
        logger.info(f"IMHK KL divergence: {kl_imhk:.6f}")
        logger.info(f"Klein KL divergence: {kl_klein:.6f}")
        
        # Compare the samplers
        if tv_distance_klein > 0:
            tv_ratio = tv_distance_imhk / tv_distance_klein
            logger.info(f"IMHK/Klein TV ratio: {tv_ratio:.4f}")
        
        if kl_klein > 0:
            kl_ratio = kl_imhk / kl_klein
            logger.info(f"IMHK/Klein KL ratio: {kl_ratio:.4f}")
        
        logger.info("Basic example completed.")
    except Exception as e:
        logger.error(f"Error in basic example: {e}")
        raise


def run_comprehensive_tests(
    dimensions: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    basis_types: Optional[List[str]] = None,
    num_samples: int = 2000,
    parallel: bool = True
) -> Dict[Tuple, Dict[str, Any]]:
    """
    Run comprehensive tests including parameter sweeps for the research paper.
    
    Purpose:
    This function systematically explores the parameter space of lattice samplers
    across multiple dimensions, Gaussian parameters (sigma), and basis types.
    It generates comparative statistics and visualizations to evaluate the
    performance and quality of IMHK versus Klein sampling.
    
    Relevance to Lattice-based Cryptography:
    Comprehensive parameter testing is crucial for:
    - Establishing optimal parameters for cryptographic implementations
    - Understanding the security-efficiency tradeoffs in lattice-based schemes
    - Validating theoretical bounds with empirical results
    - Identifying edge cases where sampling quality might degrade
    
    The results directly inform parameter selection for lattice-based signature
    and encryption schemes, helping balance security, efficiency, and implementation
    constraints.
    
    Args:
        dimensions: List of lattice dimensions to test (default: [2, 3, 4])
        sigmas: List of Gaussian parameters to test (default: [0.5, 1.0, 2.0, 3.0, 5.0])
        basis_types: List of basis types to test (default: ['identity', 'skewed', 'ill-conditioned'])
        num_samples: Number of samples for each configuration (default: 2000)
        parallel: Whether to use parallel processing (default: True)
        
    Returns:
        Dictionary of results indexed by configuration parameters
        
    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info("Running comprehensive parameter sweep...")
    
    # Input validation
    if dimensions is not None:
        for dim in dimensions:
            if not isinstance(dim, (int, Integer)) or dim < 2:
                raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    else:
        # Default dimensions
        dimensions = [2, 3, 4]
    
    if sigmas is not None:
        for sigma in sigmas:
            if not isinstance(sigma, (float, RealNumber)) or sigma <= 0:
                raise ValueError(f"Sigma must be positive, got {sigma}")
    else:
        # Default sigma values
        # Include values below and above the smoothing parameter
        sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    if basis_types is not None:
        valid_basis_types = ['identity', 'skewed', 'ill-conditioned']
        for basis_type in basis_types:
            if basis_type not in valid_basis_types:
                raise ValueError(f"Unknown basis type: {basis_type}. Valid options are {valid_basis_types}")
    else:
        # Default basis types
        basis_types = ['identity', 'skewed', 'ill-conditioned']
    
    if not isinstance(num_samples, (int, Integer)) or num_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {num_samples}")
    
    try:
        # Run the parameter sweep
        results = parameter_sweep(
            dimensions=dimensions,
            sigmas=sigmas,
            basis_types=basis_types,
            num_samples=num_samples,
            parallel=parallel
        )
        
        # Run additional analysis on convergence times
        compare_convergence_times(results)
        
        logger.info("Comprehensive tests completed.")
        return results
    except Exception as e:
        logger.error(f"Error in comprehensive tests: {e}")
        raise


def run_specific_experiment(
    dim: int = 3,
    sigma: float = 2.0,
    num_samples: int = 5000,
    burn_in: int = 2000,
    basis_type: str = 'skewed',
    center: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Run a specific experiment with detailed analysis for the research paper.
    
    Purpose:
    This function performs an in-depth analysis of lattice sampling under specific
    conditions of interest, such as a 3D lattice with a skewed basis. It generates
    detailed visualizations and statistics to analyze the behavior and quality of
    the sampling algorithms.
    
    Relevance to Lattice-based Cryptography:
    Specific experiments with carefully selected parameters help:
    - Analyze edge cases relevant to cryptographic constructions
    - Study the behavior of samplers under non-ideal conditions (e.g., skewed bases)
    - Generate figures and data for research publications
    - Validate theoretical claims with empirical evidence
    
    These detailed experiments provide insights into practical implementations of
    lattice-based cryptography, where ideal conditions are rarely met.
    
    Args:
        dim: Dimension of the lattice (default: 3)
        sigma: Standard deviation of the Gaussian (default: 2.0)
        num_samples: Number of samples to generate (default: 5000)
        burn_in: Number of initial samples to discard (default: 2000)
        basis_type: Type of lattice basis (default: 'skewed')
        center: Center of the Gaussian distribution (default: origin)
        
    Returns:
        Dictionary containing experiment results
        
    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info("Running specific experiment with detailed analysis...")
    
    # Input validation
    if not isinstance(dim, (int, Integer)) or dim < 2:
        raise ValueError(f"Dimension must be an integer ≥ 2, got {dim}")
    
    if not isinstance(sigma, (float, RealNumber)) or sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    if not isinstance(num_samples, (int, Integer)) or num_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {num_samples}")
    
    if not isinstance(burn_in, (int, Integer)) or burn_in < 0:
        raise ValueError(f"Burn-in must be non-negative, got {burn_in}")
    
    valid_basis_types = ['identity', 'skewed', 'ill-conditioned']
    if basis_type not in valid_basis_types:
        raise ValueError(f"Unknown basis type: {basis_type}. Valid options are {valid_basis_types}")
    
    if center is not None:
        if len(center) != dim:
            raise ValueError(f"Center dimension ({len(center)}) must match lattice dimension ({dim})")
    
    try:
        # Run the experiment
        result = run_experiment(
            dim=dim,
            sigma=sigma,
            num_samples=num_samples,
            basis_type=basis_type,
            compare_with_klein=True,
            center=center
        )
        
        logger.info("Specific experiment completed.")
        return result
    except Exception as e:
        logger.error(f"Error in specific experiment: {e}")
        raise


def main() -> None:
    """
    Main entry point for running lattice sampling experiments.
    
    This function creates necessary directories, sets random seeds for reproducibility,
    and executes the selected experiments.
    """
    # Create results directories if they don't exist
    try:
        Path('results/logs').mkdir(parents=True, exist_ok=True)
        Path('results/plots').mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        raise
    
    # Set random seed for reproducibility
    np.random.seed(42)
    set_random_seed(42)
    
    logger.info("Starting lattice sampling experiments")
    
    # Run basic example for quick testing
    run_basic_example()
    
    # Uncomment to run comprehensive tests
    # run_comprehensive_tests()
    
    # Uncomment to run a specific detailed experiment
    # run_specific_experiment()
    
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()