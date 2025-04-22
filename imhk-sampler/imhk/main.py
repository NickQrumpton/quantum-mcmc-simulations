import numpy as np
import os
from sage.all import *
from .samplers import imhk_sampler, klein_sampler
from .diagnostics import plot_trace, plot_autocorrelation, compute_autocorrelation
from .visualization import plot_2d_samples
from .stats import compute_total_variation_distance, compute_kl_divergence
from .experiments import parameter_sweep, compare_convergence_times, run_experiment

def run_basic_example():
    """
    Run a basic example of the IMHK sampler on a 2D lattice.
    """
    print("Running basic 2D IMHK example...")
    
    # Parameters
    dim = 2
    sigma = 2.0
    num_samples = 2000
    burn_in = 1000
    
    # Identity basis
    B = matrix.identity(RR, dim)
    
    # Run IMHK sampler
    samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
        B, sigma, num_samples, burn_in=burn_in)
    
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
    # Plot trace
    plot_trace(samples, "basic_example_trace.png", "IMHK Sample Trace (2D)")
    
    # Compute autocorrelation
    acf_by_dim = compute_autocorrelation(samples)
    plot_autocorrelation(acf_by_dim, "basic_example_acf.png", "IMHK Autocorrelation (2D)")
    
    # Plot samples
    plot_2d_samples(samples, sigma, "basic_example_samples.png", B, "IMHK Samples (2D)")
    
    # Run Klein sampler for comparison
    klein_samples = [klein_sampler(B, sigma) for _ in range(num_samples)]
    plot_2d_samples(klein_samples, sigma, "basic_example_klein.png", B, "Klein Samples (2D)")
    
    # Compute statistical distances
    tv_distance_imhk = compute_total_variation_distance(samples, sigma, B)
    tv_distance_klein = compute_total_variation_distance(klein_samples, sigma, B)
    
    print(f"IMHK Total Variation distance: {tv_distance_imhk:.6f}")
    print(f"Klein Total Variation distance: {tv_distance_klein:.6f}")
    
    # Compute KL divergence
    kl_imhk = compute_kl_divergence(samples, sigma, B)
    kl_klein = compute_kl_divergence(klein_samples, sigma, B)
    
    print(f"IMHK KL divergence: {kl_imhk:.6f}")
    print(f"Klein KL divergence: {kl_klein:.6f}")
    
    print("Basic example completed.")

def run_comprehensive_tests():
    """
    Run comprehensive tests including parameter sweeps for the paper.
    """
    print("Running comprehensive parameter sweep...")
    
    # Dimensions to test
    dimensions = [2, 3, 4]
    
    # Sigma values to test
    # Include values below and above the smoothing parameter
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Basis types to test
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    
    # Number of samples for each configuration
    num_samples = 2000
    
    # Run the parameter sweep
    results = parameter_sweep(
        dimensions=dimensions,
        sigmas=sigmas,
        basis_types=basis_types,
        num_samples=num_samples
    )
    
    # Run additional analysis
    compare_convergence_times(results)
    
    print("Comprehensive tests completed.")

def run_specific_experiment():
    """
    Run a specific experiment with detailed analysis for the paper.
    """
    print("Running specific experiment with detailed analysis...")
    
    # Parameters
    dim = 3
    sigma = 2.0
    num_samples = 5000
    burn_in = 2000
    basis_type = 'skewed'
    
    # Create a skewed basis
    B = matrix.identity(RR, dim)
    B[0, 1] = 1.5
    B[0, 2] = 0.5
    
    # Run the experiment
    result = run_experiment(
        dim=dim,
        sigma=sigma,
        num_samples=num_samples,
        basis_type=basis_type,
        compare_with_klein=True
    )
    
    print("Specific experiment completed.")

if __name__ == "__main__":
    # Create results directories if they don't exist
    os.makedirs('results/logs', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    set_random_seed(42)
    
    # Run basic example for quick testing
    run_basic_example()
    
    # Uncomment to run comprehensive tests
    # run_comprehensive_tests()
    
    # Uncomment to run a specific detailed experiment
    # run_specific_experiment()
    
    print("All experiments completed.")