"""
Cryptographically enhanced experiments module for IMHK sampler.

This module extends the basic experiments with cryptographically relevant
configurations and analysis methods, aligned with NIST standards and
current lattice-based cryptography research.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sage.all import *
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
import json

from cryptographic_config import CryptographicParameters

# Import base experiment functions
from experiments import (
    create_lattice_basis,
    calculate_smoothing_parameter,
    run_experiment as base_run_experiment
)

# Configure logging
logger = logging.getLogger("crypto_experiments")


def calculate_security_metrics(samples: np.ndarray, 
                             sigma: float,
                             lattice_basis: Matrix,
                             dimension: int) -> Dict[str, float]:
    """
    Calculate cryptographic security metrics for lattice samples.
    
    Args:
        samples: Generated samples
        sigma: Gaussian parameter
        lattice_basis: Lattice basis matrix
        dimension: Lattice dimension
        
    Returns:
        Dictionary of security metrics
    """
    from stats import compute_total_variation_distance, compute_kl_divergence
    
    # Basic statistical metrics
    tv_distance = compute_total_variation_distance(samples, sigma, lattice_basis)
    kl_divergence = compute_kl_divergence(samples, sigma, lattice_basis)
    
    # Security-specific metrics
    epsilon = 2**(-dimension)
    eta = calculate_smoothing_parameter(dimension, epsilon)
    sigma_eta_ratio = sigma / eta
    
    # Renyi divergence (relevant for security reductions)
    # This is a simplified approximation
    renyi_divergence = 2 * kl_divergence  # Approximation for order 2
    
    # Statistical distance to uniform (relevant for decision problems)
    # Simplified metric based on TV distance
    uniform_distance = min(1.0, tv_distance * np.sqrt(dimension))
    
    return {
        "tv_distance": tv_distance,
        "kl_divergence": kl_divergence,
        "sigma_eta_ratio": sigma_eta_ratio,
        "smoothing_parameter": eta,
        "renyi_divergence": renyi_divergence,
        "uniform_distance": uniform_distance,
        "security_margin": max(0, sigma_eta_ratio - 1.0)
    }


def run_crypto_experiment(dim: int,
                         sigma: float,
                         num_samples: int,
                         basis_type: str = "identity",
                         security_level: str = "standard") -> Dict[str, Any]:
    """
    Run cryptographically focused experiment with enhanced metrics.
    
    Args:
        dim: Lattice dimension
        sigma: Gaussian parameter
        num_samples: Number of samples to generate
        basis_type: Type of lattice basis
        security_level: Target security level
        
    Returns:
        Experiment results with crypto metrics
    """
    logger.info(f"Running crypto experiment: dim={dim}, σ={sigma}, security={security_level}")
    
    # Create cryptographic basis
    if basis_type in ["identity", "skewed", "ill-conditioned", "q-ary"]:
        B = CryptographicParameters.create_cryptographic_basis(dim, basis_type)
    else:
        B = create_lattice_basis(dim, basis_type)
    
    # Run base experiment
    result = base_run_experiment(
        dim=dim,
        sigma=sigma,
        num_samples=num_samples,
        basis_type=basis_type,
        compare_with_klein=True
    )
    
    # Add cryptographic metrics if successful
    if "error" not in result and "imhk_samples" in result:
        crypto_metrics = calculate_security_metrics(
            result["imhk_samples"],
            sigma,
            B,
            dim
        )
        result.update(crypto_metrics)
        
        # Add configuration info
        result["crypto_config"] = {
            "security_level": security_level,
            "basis_type": basis_type,
            "dimension": dim,
            "estimated_bit_security": CryptographicParameters.get_security_parameters(dim)["estimated_bit_security"]
        }
    
    return result


def crypto_parameter_sweep(config_type: str = "research",
                          parallel: bool = True) -> Dict[Tuple, Dict[str, Any]]:
    """
    Perform parameter sweep with cryptographic configurations.
    
    Args:
        config_type: Configuration type ("research", "crypto", "nist")
        parallel: Whether to use parallel processing
        
    Returns:
        Results dictionary
    """
    logger.info(f"Starting crypto parameter sweep with config: {config_type}")
    
    # Get configuration
    config = CryptographicParameters.get_experiment_config(config_type)
    
    results = {}
    experiment_params = []
    
    # Build experiment parameter list
    for dim in config["dimensions"]:
        sigmas = config["sigmas"](dim)
        for sigma in sigmas:
            for basis_type in config["basis_types"]:
                params = {
                    "dim": dim,
                    "sigma": sigma,
                    "num_samples": config["num_samples"],
                    "basis_type": basis_type,
                    "security_level": config_type
                }
                experiment_params.append(params)
    
    # Run experiments
    if parallel:
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            experiment_results = pool.map(
                lambda p: run_crypto_experiment(**p),
                experiment_params
            )
    else:
        experiment_results = [run_crypto_experiment(**p) for p in experiment_params]
    
    # Organize results
    for params, result in zip(experiment_params, experiment_results):
        key = (params["dim"], params["sigma"], params["basis_type"])
        results[key] = result
    
    return results


def analyze_crypto_scalability(max_dimension: int = 128) -> Dict[int, Dict[str, Any]]:
    """
    Analyze scalability of IMHK for cryptographic dimensions.
    
    Tests increasingly large dimensions to find practical limits.
    
    Args:
        max_dimension: Maximum dimension to test
        
    Returns:
        Scalability analysis results
    """
    logger.info(f"Analyzing crypto scalability up to dimension {max_dimension}")
    
    # Test dimensions in geometric progression
    dimensions = []
    dim = 8
    while dim <= max_dimension:
        dimensions.append(dim)
        dim *= 2
    
    results = {}
    
    for dim in dimensions:
        logger.info(f"Testing dimension {dim}")
        
        # Use minimal samples for scalability test
        num_samples = min(1000, 10000 // dim)  # Scale down with dimension
        
        # Test with standard security parameter
        sigma = CryptographicParameters.get_sigma_values(dim, [2.0])[0]
        
        start_time = time.time()
        
        try:
            result = run_crypto_experiment(
                dim=dim,
                sigma=sigma,
                num_samples=num_samples,
                basis_type="identity",
                security_level="scalability_test"
            )
            
            elapsed_time = time.time() - start_time
            
            # Add timing information
            result["computation_time"] = elapsed_time
            result["samples_per_second"] = num_samples / elapsed_time
            
            results[dim] = result
            
        except Exception as e:
            logger.error(f"Failed at dimension {dim}: {e}")
            results[dim] = {
                "error": str(e),
                "dimension": dim,
                "max_practical_dimension": dimensions[dimensions.index(dim) - 1] if dim > 8 else 8
            }
            break
    
    return results


def run_lattice_attack_simulation(dimension: int,
                                 sigma: float,
                                 attack_type: str = "bkz") -> Dict[str, Any]:
    """
    Simulate lattice reduction attacks to validate security parameters.
    
    Args:
        dimension: Lattice dimension
        sigma: Gaussian parameter
        attack_type: Type of attack to simulate
        
    Returns:
        Attack simulation results
    """
    logger.info(f"Simulating {attack_type} attack on {dimension}-dim lattice")
    
    # This is a simplified simulation for research purposes
    # Real attacks would use lattice reduction algorithms
    
    results = {
        "dimension": dimension,
        "sigma": sigma,
        "attack_type": attack_type
    }
    
    # Estimate attack complexity based on dimension
    if attack_type == "bkz":
        # BKZ attack complexity estimation
        block_size = min(dimension, 50)  # Practical BKZ block size
        operations = 2**(0.292 * block_size)  # Simplified BKZ complexity
        
        results["estimated_operations"] = operations
        results["estimated_time_seconds"] = operations / 1e9  # Assuming 1 GHz attacker
        results["security_bits"] = np.log2(operations)
        
    elif attack_type == "primal":
        # Primal lattice attack estimation
        results["security_bits"] = 0.265 * dimension  # Simplified estimate
        
    # Compare with chosen parameters
    sigma_eta_ratio = sigma / calculate_smoothing_parameter(dimension)
    results["sigma_eta_ratio"] = sigma_eta_ratio
    results["secure"] = sigma_eta_ratio > 1.0 and results["security_bits"] > 80
    
    return results


def generate_crypto_benchmark_report(all_results: Dict[str, Any],
                                   output_dir: Path) -> None:
    """
    Generate comprehensive benchmark report for cryptographic applications.
    
    Args:
        all_results: Combined experiment results
        output_dir: Directory for output files
    """
    logger.info("Generating crypto benchmark report")
    
    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "IMHK Cryptographic Sampler",
        "configurations_tested": {},
        "performance_metrics": {},
        "security_analysis": {},
        "recommendations": []
    }
    
    # Analyze configurations
    for config_type in ["research", "crypto", "nist"]:
        if config_type in all_results:
            config_results = all_results[config_type]
            report["configurations_tested"][config_type] = {
                "dimensions": list(set(key[0] for key in config_results.keys())),
                "sigmas": list(set(key[1] for key in config_results.keys())),
                "basis_types": list(set(key[2] for key in config_results.keys()))
            }
    
    # Performance metrics
    if "scalability" in all_results:
        scalability = all_results["scalability"]
        max_dim = max(d for d in scalability.keys() if "error" not in scalability[d])
        report["performance_metrics"]["max_practical_dimension"] = max_dim
        report["performance_metrics"]["throughput_by_dimension"] = {
            d: scalability[d].get("samples_per_second", 0)
            for d in scalability.keys()
            if "error" not in scalability[d]
        }
    
    # Security analysis
    report["security_analysis"]["recommended_parameters"] = {
        "minimum_sigma_eta_ratio": 2.0,
        "standard_dimensions": [32, 64, 128],
        "secure_basis_types": ["identity", "skewed"]
    }
    
    # Generate recommendations
    report["recommendations"] = [
        "Use dimension ≥ 32 for cryptographic applications",
        "Maintain σ/η ratio ≥ 2.0 for security margin",
        "Consider dimension 64 for standard security level",
        "IMHK performs well on skewed lattices common in crypto"
    ]
    
    # Save report
    with open(output_dir / "crypto_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info("Benchmark report generated")


# Plotting functions for cryptographic analysis
def plot_crypto_security_landscape(results: Dict[str, Any],
                                 output_path: Path) -> None:
    """
    Create security landscape visualization for crypto parameters.
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap of security metrics
    dimensions = sorted(set(key[0] for key in results.keys()))
    sigmas = sorted(set(key[1] for key in results.keys()))
    
    security_matrix = np.zeros((len(dimensions), len(sigmas)))
    
    for i, dim in enumerate(dimensions):
        for j, sigma in enumerate(sigmas):
            key = (dim, sigma, "identity")
            if key in results and "security_margin" in results[key]:
                security_matrix[i, j] = results[key]["security_margin"]
    
    im = plt.imshow(security_matrix, aspect='auto', cmap='RdYlGn')
    plt.colorbar(im, label='Security Margin (σ/η - 1)')
    
    plt.xticks(range(len(sigmas)), [f"{s:.1f}" for s in sigmas])
    plt.yticks(range(len(dimensions)), dimensions)
    plt.xlabel('Gaussian Parameter σ')
    plt.ylabel('Lattice Dimension')
    plt.title('Cryptographic Security Landscape')
    
    # Add security threshold line
    threshold_line = np.ones_like(security_matrix) * 1.0
    plt.contour(threshold_line, levels=[1.0], colors='red', linewidths=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_vs_security(results: Dict[str, Any],
                               output_path: Path) -> None:
    """
    Plot performance metrics against security parameters.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    dimensions = []
    acceptance_rates = []
    security_bits = []
    tv_distances = []
    
    for key, data in results.items():
        if "error" not in data:
            dim = key[0]
            dimensions.append(dim)
            acceptance_rates.append(data.get("imhk_acceptance_rate", 0))
            
            # Estimate security bits
            sec_params = CryptographicParameters.get_security_parameters(dim)
            security_bits.append(sec_params["estimated_bit_security"])
            
            tv_distances.append(data.get("tv_distance", 1.0))
    
    # Plot 1: Acceptance rate vs security
    ax1.scatter(security_bits, acceptance_rates, s=100, alpha=0.7)
    ax1.set_xlabel('Estimated Security (bits)')
    ax1.set_ylabel('Acceptance Rate')
    ax1.set_title('IMHK Efficiency vs Security Level')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(security_bits, acceptance_rates, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(security_bits), max(security_bits), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Plot 2: Quality vs dimension
    ax2.scatter(dimensions, tv_distances, s=100, alpha=0.7)
    ax2.set_xlabel('Lattice Dimension')
    ax2.set_ylabel('Total Variation Distance')
    ax2.set_yscale('log')
    ax2.set_title('Sampling Quality vs Dimension')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()