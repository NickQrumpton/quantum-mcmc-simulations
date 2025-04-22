#!/usr/bin/env sage
"""
Core Functions for IMHK Algorithm
---------------------------------
This module contains core mathematical functions for the IMHK algorithm.

EXAMPLES::

    >>> from imhk.core import discrete_gaussian_pdf, rho_function, \
    ...     sum_rho_over_integers, discrete_gaussian_sampler_1d

    >>> discrete_gaussian_pdf(0, 1.0)
    1.0
    >>> discrete_gaussian_pdf([1, 0], 1.0)
    0.6065306597126334
    >>> from sage.all import vector, ZZ, RR
    >>> discrete_gaussian_pdf(vector(ZZ, [1, 1]), 1.0)
    0.36787944117144233
    >>> discrete_gaussian_pdf(vector(RR, [0, 0]), 2.0, [1, 1])
    0.8824969025845955
    >>> try:
    ...     discrete_gaussian_pdf([1, 2], 0)
    ... except ValueError:
    ...     print("Caught sigma error")
    Caught sigma error
"""

import numpy as np
from typing import Union, List, Optional

def discrete_gaussian_pdf(
    x: Union[float, List[float], np.ndarray, 'FreeModuleElement'],
    sigma: float,
    center: Optional[Union[List[float], np.ndarray, 'FreeModuleElement']] = None
) -> float:
    if sigma <= 0:
        raise ValueError(f"Standard deviation must be positive, got sigma={sigma}")

    if isinstance(x, (int, float)):
        x_np = np.array([float(x)])
    elif str(type(x)).find('sage.modules.vector') >= 0 or hasattr(x, 'list'):
        x_np = np.array([float(xi) for xi in x])
    elif isinstance(x, (list, tuple, np.ndarray)):
        x_np = np.array(x, dtype=float)
    else:
        raise TypeError(f"Unsupported type for x: {type(x)}")

    if center is None:
        center_np = np.zeros_like(x_np)
    elif str(type(center)).find('sage.modules.vector') >= 0 or hasattr(center, 'list'):
        center_np = np.array([float(ci) for ci in center])
    elif isinstance(center, (list, tuple, np.ndarray)):
        center_np = np.array(center, dtype=float)
    else:
        raise TypeError(f"Unsupported type for center: {type(center)}")

    if x_np.shape != center_np.shape:
        raise ValueError(f"Shape mismatch: x has shape {x_np.shape}, center has shape {center_np.shape}")

    squared_dist = np.sum((x_np - center_np) ** 2)
    return float(np.exp(-squared_dist / (2 * sigma**2)))

def rho_function(
    x: Union[float, List[float], np.ndarray],
    sigma: float,
    center: Optional[Union[float, List[float], np.ndarray]] = None
) -> float:
    if center is None:
        center = 0.0 if isinstance(x, (int, float)) else np.zeros_like(x)

    if isinstance(x, (int, float)) and isinstance(center, (int, float)):
        squared_dist = (x - center) ** 2
    else:
        x_np = np.array(x, dtype=float)
        center_np = np.array(center, dtype=float)
        squared_dist = np.sum((x_np - center_np) ** 2)

    return float(np.exp(-np.pi * squared_dist / sigma**2))

def sum_rho_over_integers(sigma: float, center: float) -> float:
    trunc = int(np.ceil(6 * sigma))
    lower, upper = int(np.floor(center - trunc)), int(np.ceil(center + trunc))
    return sum(rho_function(k, sigma, center) for k in range(lower, upper + 1))

def discrete_gaussian_sampler_1d(center: float, sigma: float) -> int:
    if sigma is None or sigma != sigma or sigma <= 1e-10:
        import warnings
        warnings.warn(f"[patched] Invalid sigma={sigma}; clamping to 0.01")
        sigma = 0.01

    trunc = int(np.ceil(6 * sigma))
    lower = int(np.floor(center - trunc))
    upper = int(np.ceil(center + trunc))
    if lower >= upper:
        upper = lower + 1

    probs = np.array([rho_function(k, sigma, center) for k in range(lower, upper + 1)], dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"Probability mass underflow (sum={total})")
    probs /= total

    values = np.arange(lower, upper + 1)
    return int(np.random.choice(values, p=probs))

# ✅ Doctest setup for module testing
import sys
_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {
        name: obj for name, obj in list(locals().items())
        if callable(obj) and obj.__doc__ and '>>>' in obj.__doc__
    }
