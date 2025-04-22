#!/usr/bin/env sage
r"""
Independent Metropolis-Hastings-Klein (IMHK) Algorithm
"""

import logging
from sage.all import vector
from classical_sampler.imhk.core import (
    discrete_gaussian_pdf,
    discrete_gaussian_sampler_1d
)

logger = logging.getLogger("IMHK_Sampler")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

MIN_SIGMA_I = 0.01

def gram_schmidt(B):
    n = B.nrows()
    GS = []
    mu = [[0] * n for _ in range(n)]

    for i in range(n):
        vi = B.row(i).change_ring(B.base_ring())
        projection = vector(B.base_ring(), [0] * B.ncols())
        for j in range(i):
            denom = GS[j] * GS[j]
            if denom < 1e-12:
                logger.warning(f"Degenerate GS vector at index {j}")
                continue
            mu[i][j] = vi * GS[j] / denom
            projection += mu[i][j] * GS[j]
        gs_vec = vi - projection
        if gs_vec.norm() < 1e-12:
            logger.warning(f"Replacing GS[{i}] with fallback vector")
            from random import uniform
            gs_vec = vector(B.base_ring(), [uniform(-1e-3, 1e-3) for _ in range(B.ncols())])
        GS.append(gs_vec)
        mu[i][i] = 1
    return GS, mu

def klein_sampler(B, sigma):
    GS, mu = gram_schmidt(B)
    n = B.nrows()
    z = [0] * n
    for i in reversed(range(n)):
        norm = GS[i].norm()
        if norm <= 1e-10 or norm != norm:
            logger.warning(f"GS[{i}].norm() invalid ({norm}), skipping")
            z[i] = 0
            continue
        sigma_i = max(sigma / norm, 0.5)  # ← more robust
        c = sum(mu[i][j] * z[j] for j in range(i + 1, n))
        try:
            z[i] = discrete_gaussian_sampler_1d(sigma_i, -c)
        except Exception as e:
            logger.error(f"Sampling failed at index {i}: {e}")
            z[i] = 0
    return vector(B.base_ring(), z), B * vector(z)

def imhk_step(B, x, sigma):
    from random import random
    z, y = klein_sampler(B, sigma)
    try:
        ratio = discrete_gaussian_pdf(y, sigma) / discrete_gaussian_pdf(x, sigma)
    except ZeroDivisionError:
        logger.warning("Zero division in acceptance ratio computation; rejecting move.")
        return x
    p_accept = min(1, ratio)
    return y if random() < p_accept else x

def imhk_sampler(B, sigma, num_samples=1000, x0=None):
    if x0 is None:
        x0 = vector(B.base_ring(), [0] * B.ncols())
    samples = []
    x = x0
    for _ in range(num_samples):
        x = imhk_step(B, x, sigma)
        samples.append(x)
    return samples

def parallel_imhk_sampler(*args, **kwargs):
    raise NotImplementedError("Parallel sampler is not implemented in this module")

# ✅ Safe doctest discovery
import sys
_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {
        name: obj for name, obj in locals().items()
        if callable(obj) and obj.__doc__ and '>>>' in obj.__doc__
    }
