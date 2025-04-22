#!/usr/bin/env sage
"""
Experiment Functions for IMHK Algorithm
--------------------------------------
This module contains functions for running experiments with the IMHK algorithm.
"""

import os
import time
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from sage.all import identity_matrix, ZZ, RR, vector

from .core import discrete_gaussian_pdf
from .sampler import klein_sampler, imhk_sampler
from .diagnostics import compute_total_variation_distance
from .lattice import generate_random_lattice, truncate_lattice

# ✅ Local logger (DO NOT import from imhk to avoid circular import)
import logging
logger = logging.getLogger("IMHK_Sampler")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def validate_sampler(B, sigma, num_samples, quick_mode=False):
    """
    Run the sampler and return acceptance statistics.
    """
    try:
        samples = imhk_sampler(B, sigma, num_samples)
        return {
            "num_samples": len(samples),
            "acceptance_rate": "N/A (sequential fallback mode)"
        }
    except Exception as e:
        logger.error(f"Validation sampler failed: {e}")
        return {"error": str(e)}

def run_high_dimensional_test(quick_mode=False):
    """
    Run a test on a high-dimensional lattice with IMHK sampling (non-parallel fallback).
    """
    logger.info("Running high-dimensional test")

    if quick_mode:
        dimension = 4
        sigma = 5.0
        num_samples = 10
        burn_in = 2
        num_chains = 1
    else:
        dimension = 8
        sigma = 10.0
        num_samples = 50
        burn_in = 20
        num_chains = min(4, mp.cpu_count())

    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory usage before test: {memory_before:.2f} MB")
    except ImportError:
        logger.warning("psutil not installed. Memory usage tracking skipped.")
        memory_before = None

    try:
        B = generate_random_lattice(dimension)
        bounds = truncate_lattice(B, sigma)
        logger.info("Truncation bounds determined")

        start_time = time.time()
        logger.info(f"Sampling with {dimension}D lattice: samples={num_samples}, burn_in={burn_in}, chains={num_chains}")

        # ✅ FALLBACK: use imhk_sampler sequentially instead of parallel sampler
        samples = []
        for _ in range(num_chains):
            s = imhk_sampler(B, sigma, num_samples)
            samples.extend(s)
        samples_x = samples
        samples_z = []
        acceptance_rate = "N/A (sequential mode)"
        runtime = time.time() - start_time

        logger.info(f"Test complete in {runtime:.2f} seconds")
        logger.info(f"{len(samples_x)} samples generated")

        results = {
            'dimension': dimension,
            'sigma': sigma,
            'acceptance_rate': acceptance_rate,
            'runtime': runtime,
            'num_samples': len(samples_x)
        }

        if memory_before is not None:
            try:
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_used = memory_after - memory_before
                logger.info(f"Memory usage after: {memory_after:.2f} MB (+{memory_used:.2f} MB)")
                results['memory_usage_mb'] = memory_after
                results['memory_increase_mb'] = memory_used
            except Exception as e:
                logger.error(f"Memory tracking failed: {e}")

        if not quick_mode:
            try:
                os.makedirs("results/data", exist_ok=True)
                with open("results/data/high_dimensional_test.pkl", 'wb') as f:
                    pickle.dump(results, f)
                logger.info("Results saved to results/data/high_dimensional_test.pkl")
            except Exception as e:
                logger.error(f"Could not save results: {e}")

        return results

    except Exception as e:
        logger.error(f"High-dimensional test failed: {e}")
        return None

# ✅ Doctest discovery
import sys
_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {
        name: obj for name, obj in list(locals().items())
        if callable(obj) and isinstance(obj.__doc__, str) and '>>>' in obj.__doc__
    }
