#!/usr/bin/env sage
"""
Independent Metropolis-Hastings-Klein (IMHK) Algorithm
------------------------------------------------------
A package for sampling from discrete Gaussian distributions over lattices
using the IMHK algorithm.

EXAMPLES::

    >>> from imhk.core import discrete_gaussian_pdf, rho_function, \
    ...     sum_rho_over_integers, discrete_gaussian_sampler_1d
    >>> from imhk.sampler import klein_sampler, imhk_step, \
    ...     imhk_sampler, parallel_imhk_sampler
    >>> from imhk.diagnostics import compute_total_variation_distance, \
    ...     estimate_mixing_time
    >>> from imhk.lattice import generate_random_lattice, generate_skewed_lattice, \
    ...     generate_ill_conditioned_lattice, generate_ntru_lattice, \
    ...     apply_lattice_reduction, truncate_lattice
    >>> from imhk.experiments import run_high_dimensional_test, validate_sampler
    >>> discrete_gaussian_pdf(0, 1.0)
    1.0
"""

# --- Core functionality ---
from .core import (
    discrete_gaussian_pdf, rho_function, sum_rho_over_integers, discrete_gaussian_sampler_1d
)

# --- Sampling algorithms ---
from .sampler import (
    klein_sampler, imhk_step, imhk_sampler, parallel_imhk_sampler
)

# --- Diagnostics and convergence tools ---
from .diagnostics import (
    compute_total_variation_distance,
    estimate_mixing_time,
    # Other diagnostics are not yet implemented:
    # compute_kl_divergence,
    # compute_log_likelihood,
    # compute_moments,
    # compute_ess,
    # compute_gelman_rubin,
    # compute_geweke,
    # compute_autocorrelation,
    # compute_theoretical_spectral_bound
)

# --- Lattice utilities ---
from .lattice import (
    generate_random_lattice, generate_skewed_lattice, generate_ill_conditioned_lattice,
    generate_ntru_lattice, apply_lattice_reduction, truncate_lattice
)

# --- Experiment scripts ---
from .experiments import (
    run_high_dimensional_test, validate_sampler
)

# --- Logging configuration ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("IMHK_Sampler")

# --- Safe doctest discovery setup ---
import sys

_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {}

    for name, obj in list(locals().items()):
        if callable(obj):
            doc = getattr(obj, '__doc__', None)
            if isinstance(doc, str) and '>>>' in doc:
                __test__[name] = obj
