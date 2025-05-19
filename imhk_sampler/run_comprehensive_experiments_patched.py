#!/usr/bin/env sage -python
"""
Patched run_comprehensive_experiments function with proper type handling.
"""

import numpy as np
import pandas as pd
import logging
import time
import traceback
from math import sqrt
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_value_for_format(value, default=0.0):
    """Ensure value is not None and can be formatted."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_comprehensive_experiments_patched(self):
    """Run all experiments with complete type checking and error handling."""
    logger.info("Starting comprehensive experiments (PATCHED)")
    
    # Import required modules with proper fallbacks
    try:
        from utils import create_lattice_basis
        from parameter_config import compute_smoothing_parameter
        from diagnostics import compute_ess
        from fixed_samplers_v2_patched import fixed_imhk_sampler_wrapper, fixed_klein_sampler_wrapper
        from fixed_tv_distance_calculation_v2 import compute_tv_distance_structured
    except ImportError as e:
        logger.error(f"Import error: {e}")
        # Try alternative imports
        try:
            from utils import create_lattice_basis
            from parameter_config import compute_smoothing_parameter
            from diagnostics import compute_ess
            from fixed_samplers_v2 import fixed_imhk_sampler_wrapper, fixed_klein_sampler_wrapper
            from fixed_tv_distance_calculation import compute_tv_distance_structured
        except ImportError as e2:
            logger.error(f"Alternative import also failed: {e2}")
            raise
    
    # Define experiment configurations with proper type handling
    experiments = {
        'standard_lattices': [
            # (dimension, basis_type, sigma_ratios, num_samples)
            (4, 'identity', np.linspace(0.5, 3.0, 8), 5000),
            (8, 'identity', np.linspace(0.5, 3.0, 6), 3000),
            (16, 'identity', np.linspace(0.5, 2.5, 5), 2000),
            (32, 'identity', np.linspace(1.0, 2.0, 4), 1000),
        ],
        'cryptographic_lattices': [
            # Q-ary lattices
            (8, 'q-ary', np.linspace(0.8, 2.5, 5), 3000),
            (16, 'q-ary', np.linspace(0.8, 2.0, 4), 2000),
            (32, 'q-ary', np.linspace(1.0, 1.8, 3), 1000),
            # NTRU and PrimeCyclotomic
            (512, 'NTRU', np.linspace(0.9, 1.3, 3), 500),
            (683, 'PrimeCyclotomic', np.linspace(0.9, 1.3, 3), 500),
        ]
    }
    
    all_results = []
    
    for experiment_type, configs in experiments.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {experiment_type} experiments")
        logger.info('='*60)
        
        for dim, basis_type, sigma_ratios, num_samples in configs:
            # Ensure proper types
            dim = int(dim)
            num_samples = int(num_samples)
            
            logger.info(f"\nTesting {basis_type} (dimension={dim})")
            
            # Create basis with error handling
            try:
                basis_info = create_lattice_basis(dim, basis_type)
                
                # Handle different basis types
                if isinstance(basis_info, tuple):
                    poly_mod, q = basis_info
                    # Ensure q is numeric
                    if isinstance(q, str):
                        q = int(q)
                    q = int(q)
                    
                    # Conservative sigma for structured lattices
                    base_sigma = float(sqrt(q) / 30)
                    sigmas = [float(base_sigma * ratio) for ratio in sigma_ratios]
                    eta = float(base_sigma)
                    logger.info(f"Structured lattice: q={q}, base_sigma={base_sigma:.4f}")
                else:
                    eta = float(compute_smoothing_parameter(basis_info))
                    sigmas = [float(eta * ratio) for ratio in sigma_ratios]
                    base_sigma = float(eta)
                    logger.info(f"Standard lattice: eta={eta:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to create basis for {basis_type}: {e}")
                continue
            
            for i, sigma in enumerate(sigmas):
                sigma_ratio = float(sigma_ratios[i])
                sigma = float(sigma)
                
                logger.info(f"\nExperiment: {basis_type} d={dim} σ/η={sigma_ratio:.2f}")
                
                self.metrics_tracker['experiments_run'] += 1
                
                # Initialize result dictionary with all fields
                result = {
                    'experiment_type': experiment_type,
                    'basis_type': basis_type,
                    'dimension': int(dim),
                    'sigma': float(sigma),
                    'sigma_ratio': float(sigma_ratio),
                    'eta': float(eta),
                    'num_samples': int(num_samples),
                    'imhk_acceptance_rate': 0.0,
                    'imhk_time': 0.0,
                    'klein_time': 0.0,
                    'speedup': 0.0,
                    'ess': 0.0,
                    'ess_per_second': 0.0,
                    'tv_distance': None,
                    'tv_distance_klein': None,
                    'error': None
                }
                
                try:
                    # Run IMHK sampler
                    start_time = time.time()
                    imhk_samples, imhk_metadata = fixed_imhk_sampler_wrapper(
                        basis_info=basis_info,
                        sigma=sigma,
                        num_samples=num_samples,
                        burn_in=min(1000, num_samples//2),
                        basis_type=basis_type
                    )
                    imhk_time = float(time.time() - start_time)
                    
                    result['imhk_time'] = imhk_time
                    result['imhk_acceptance_rate'] = float(imhk_metadata.get('acceptance_rate', 0.0))
                    
                    logger.info(f"IMHK: {imhk_samples.shape[0]} samples, "
                              f"acceptance={result['imhk_acceptance_rate']:.4f}, "
                              f"time={imhk_time:.2f}s")
                    
                    # Run Klein sampler for comparison
                    klein_start = time.time()
                    klein_samples, klein_metadata = fixed_klein_sampler_wrapper(
                        basis_info=basis_info,
                        sigma=sigma,
                        num_samples=min(500, num_samples//2),
                        basis_type=basis_type
                    )
                    klein_time = float(time.time() - klein_start)
                    
                    result['klein_time'] = klein_time
                    result['speedup'] = float(klein_time / imhk_time) if imhk_time > 0 else 0.0
                    
                    logger.info(f"Klein: {klein_samples.shape[0]} samples, time={klein_time:.2f}s")
                    
                    # Compute ESS with robust error handling
                    if imhk_samples.shape[0] > 0:
                        ess = self.calculate_metric_with_fallback(
                            compute_ess, imhk_samples[:, 0], 
                            default=0.0, metric_name="ESS"
                        )
                        result['ess'] = max(0.0, float(ess))
                        result['ess_per_second'] = float(result['ess'] / imhk_time) if imhk_time > 0 else 0.0
                    
                    # Compute TV distance
                    max_samples_for_tv = min(2000, imhk_samples.shape[0])
                    if max_samples_for_tv > 0:
                        # IMHK TV distance
                        tv_imhk = self.calculate_metric_with_fallback(
                            compute_tv_distance_structured,
                            imhk_samples[:max_samples_for_tv],
                            sigma, basis_info, basis_type,
                            default=None, metric_name="TV distance (IMHK)"
                        )
                        result['tv_distance'] = tv_imhk
                        
                        # Klein TV distance
                        max_klein_samples = min(500, klein_samples.shape[0])
                        if max_klein_samples > 0:
                            tv_klein = self.calculate_metric_with_fallback(
                                compute_tv_distance_structured,
                                klein_samples[:max_klein_samples],
                                sigma, basis_info, basis_type,
                                default=None, metric_name="TV distance (Klein)"
                            )
                            result['tv_distance_klein'] = tv_klein
                    
                    self.metrics_tracker['experiments_successful'] += 1
                    
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    logger.debug(traceback.format_exc())
                    result['error'] = str(e)
                    self.metrics_tracker['experiments_failed'] += 1
                
                all_results.append(result)
                
                # Log result summary with proper formatting
                acceptance_rate = sanitize_value_for_format(result['imhk_acceptance_rate'])
                ess_value = sanitize_value_for_format(result['ess'])
                tv_value = result['tv_distance'] if result['tv_distance'] is not None else 'N/A'
                
                logger.info(f"Result summary: acceptance={acceptance_rate:.4f}, "
                          f"ESS={ess_value:.2f}, TV={tv_value}")
    
    # Create DataFrame and save results
    logger.info(f"\nExperiment summary: {self.metrics_tracker['experiments_successful']} "
               f"successful, {self.metrics_tracker['experiments_failed']} failed")
    
    df = pd.DataFrame(all_results)
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['sigma', 'sigma_ratio', 'eta', 'imhk_acceptance_rate', 
                      'imhk_time', 'klein_time', 'speedup', 'ess', 'ess_per_second']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Save results with proper error handling
    try:
        csv_path = self.data_dir / 'all_results.csv'
        df.to_csv(csv_path, index=False)
        self.metrics_tracker['files_generated'].append(str(csv_path))
        logger.info(f"Saved CSV results to {csv_path}")
        
        json_path = self.data_dir / 'all_results.json'
        # Import save_json_safely
        from json_serialization_utils import save_json_safely
        save_json_safely(all_results, json_path)
        self.metrics_tracker['files_generated'].append(str(json_path))
        logger.info(f"Saved JSON results to {json_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return df