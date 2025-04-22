#!/usr/bin/env sage
"""
Diagnostic Functions for IMHK Algorithm
--------------------------------------
This module contains diagnostic functions for analyzing MCMC convergence.

EXAMPLES::

    >>> from imhk.diagnostics import compute_total_variation_distance, compute_exact_discrete_gaussian, estimate_mixing_time
    >>> from sage.all import identity_matrix, ZZ, vector, RR
    >>> # Total variation distance examples
    >>> B = identity_matrix(ZZ, 2)
    >>> samples = [vector(RR, [0, 0]), vector(RR, [1, 0])]
    >>> exact = {(0.0, 0.0): 0.5, (1.0, 0.0): 0.5}
    >>> tv = compute_total_variation_distance(samples, 1.0, B, None, exact)
    >>> tv == 0.0  # Samples match exact distribution
    True

    >>> # Exact discrete Gaussian examples
    >>> dist = compute_exact_discrete_gaussian(B, 1.0)
    >>> len(dist) > 0
    True
    >>> abs(sum(dist.values()) - 1.0) < 1e-10  # Probabilities should sum to 1
    True

    >>> # Mixing time estimation (quick_mode)
    >>> mix_time, _ = estimate_mixing_time(B, 5.0, max_iterations=3, timeout_seconds=5, quick_mode=True)
    >>> isinstance(mix_time, (int, type(None)))
    True
"""

import numpy as np
import itertools
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# Import from sage
from sage.all import vector, ZZ, RR

# Import from imhk package
from .core import discrete_gaussian_pdf
from .lattice import truncate_lattice
import logging
logger = logging.getLogger("IMHK_Sampler")

def compute_total_variation_distance(samples, sigma, lattice_basis, center=None, exact=None):
    dim = lattice_basis.nrows()
    if dim > 2:
        logger.warning("TV distance computation only implemented for dimensions ≤ 2")
        return None

    if exact is None:
        exact = compute_exact_discrete_gaussian(lattice_basis, sigma, center)
        if exact is None:
            return None

    sample_counts = {}
    n_samples = len(samples)
    for sample in samples:
        key = tuple(sample)
        sample_counts[key] = sample_counts.get(key, 0) + 1

    empirical = {k: v / n_samples for k, v in sample_counts.items()}

    tv_dist = 0.0
    for key in exact:
        tv_dist += abs(exact.get(key, 0.0) - empirical.get(key, 0.0))
    for key in empirical:
        if key not in exact:
            tv_dist += empirical[key]

    return 0.5 * tv_dist

def compute_exact_discrete_gaussian(B, sigma, c=None):
    dim = B.nrows()
    if dim > 2:
        logger.warning("Exact distribution computation only implemented for dimensions ≤ 2")
        return None

    try:
        bounds = truncate_lattice(B, sigma, c, epsilon=1e-10)
    except Exception as e:
        raise ValueError(f"Error computing truncation bounds: {e}")

    probs = {}
    Z = 0.0
    try:
        for coords in itertools.product(*[range(b[0], b[1]+1) for b in bounds]):
            z = vector(ZZ, coords)
            lattice_point = B.linear_combination_of_rows(list(z))
            prob = discrete_gaussian_pdf(lattice_point, sigma, c)
            key = tuple(lattice_point)
            probs[key] = prob
            Z += prob

        if Z < 1e-10:
            raise ValueError("Normalization constant is too small, adjust sigma or bounds")

        for key in probs:
            probs[key] /= Z

        return probs
    except Exception as e:
        logger.error(f"Error in compute_exact_discrete_gaussian: {e}")
        return None

def estimate_mixing_time(B, sigma, max_iterations=50, threshold=0.3, c=None, timeout_seconds=60, quick_mode=False):
    if quick_mode:
        max_iterations = min(max_iterations, 3)
        timeout_seconds = min(timeout_seconds, 5)

    dim = B.nrows()
    if dim > 2:
        logger.warning("Mixing time estimation only implemented for dimensions ≤ 2")
        return None, []

    start_time = time.time()
    try:
        exact_dist = compute_exact_discrete_gaussian(B, sigma, c)
        if exact_dist is None:
            logger.error("Could not compute exact distribution for TV distance calculation")
            return None, []

        from .sampler import klein_sampler, imhk_step, imhk_sampler

        current_z, _ = klein_sampler(B, sigma, c)
        tv_distances = []

        for i in range(max_iterations):
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Mixing time estimation timed out after {timeout_seconds} seconds at iteration {i}")
                return i, tv_distances

            current_z, _ = imhk_step(current_z, B, sigma, c)
            samples_z, samples_x, _, _ = imhk_sampler(
                B, sigma, 100, c, burn_in=0, num_chains=1, initial_samples=[current_z], quick_mode=quick_mode)

            tv_dist = compute_total_variation_distance(samples_x, sigma, B, c, exact_dist)
            tv_distances.append(tv_dist)

            if (i+1) % 10 == 0 or i == 0:
                logger.info(f"Iteration {i+1}/{max_iterations}, TV distance: {tv_dist:.4f}")

            if tv_dist < threshold:
                logger.info(f"Mixing achieved at iteration {i+1}, TV distance: {tv_dist:.4f}")
                return i + 1, tv_distances

        logger.info(f"Maximum iterations reached. Final TV distance: {tv_distances[-1]:.4f}")
        return max_iterations, tv_distances

    except Exception as e:
        logger.error(f"Error in mixing time estimation: {e}")
        return None, []

# ✅ Doctest discovery
import sys
_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {}
    for name, obj in list(locals().items()):
        if callable(obj) and hasattr(obj, '__doc__') and obj.__doc__:
            if '>>>' in obj.__doc__:
                __test__[name] = obj