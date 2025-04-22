#!/usr/bin/env sage
"""
Lattice Generation Functions for IMHK Algorithm
----------------------------------------------
This module contains functions for generating and manipulating lattices.

EXAMPLES::

    >>> from sage.all import ZZ
    >>> B = generate_random_lattice(2)
    >>> B.base_ring() == ZZ
    True
"""

import numpy as np
from sage.all import (
    matrix, ZZ, GF, randint, identity_matrix, diagonal_matrix, copy, random_matrix, QQ
)


def generate_random_lattice(dim, bit_size=10, det_range=(50, 100)):
    """
    Generate a random integer lattice basis.
    """
    U = random_matrix(ZZ, dim, dim)
    while abs(U.det()) != 1:
        U = random_matrix(ZZ, dim, dim)

    det = randint(det_range[0], det_range[1])
    D = diagonal_matrix([1] * (dim-1) + [det])

    V = random_matrix(ZZ, dim, dim)
    while abs(V.det()) != 1:
        V = random_matrix(ZZ, dim, dim)

    B = U * D * V

    if bit_size > 0:
        max_value = 2**bit_size - 1
        for i in range(dim):
            for j in range(dim):
                B[i, j] = max(-max_value, min(max_value, B[i, j]))

    return B


def generate_skewed_lattice(dim, skew_factor=100):
    """
    Generate a skewed lattice basis with varying vector lengths.
    """
    B = identity_matrix(ZZ, dim)
    B[0] *= skew_factor
    for i in range(1, dim):
        B[0, i] = randint(1, skew_factor // 2)
    return B


def generate_ill_conditioned_lattice(dim, condition_number=1000):
    """
    Generate an ill-conditioned lattice basis.
    """
    B = random_matrix(ZZ, dim, dim)
    while B.rank() < dim:
        B = random_matrix(ZZ, dim, dim)

    U, S, Vt = np.linalg.svd(np.array(B, dtype=float))
    S[0] = condition_number**(1/(dim-1))
    for i in range(1, dim):
        S[i] = S[0] * (1/condition_number)**(i/(dim-1))
    B_new = np.dot(U * S, Vt)

    B_int = matrix(ZZ, dim, dim)
    for i in range(dim):
        for j in range(dim):
            B_int[i, j] = int(round(B_new[i, j]))

    while B_int.rank() < dim:
        B_int += identity_matrix(ZZ, dim)

    return B_int


def generate_ntru_lattice(n, q):
    """
    Generate an NTRU lattice basis.
    """
    if n % 2 != 0:
        raise ValueError("n must be even for NTRU lattices")

    dim = 2 * n

    f = [randint(-1, 1) for _ in range(n)]
    while f.count(0) > n // 3:
        f = [randint(-1, 1) for _ in range(n)]

    g = [randint(-1, 1) for _ in range(n)]
    while g.count(0) > n // 3:
        g = [randint(-1, 1) for _ in range(n)]

    Rq = GF(q)['x']
    x = Rq.gen()
    f_poly = sum(Rq(f[i]) * x**i for i in range(n))
    g_poly = sum(Rq(g[i]) * x**i for i in range(n))
    mod_poly = x**n - 1

    try:
        f_inv = f_poly.inverse_mod(mod_poly)
        h_poly = (g_poly * f_inv) % mod_poly
        h = [int(h_poly[i]) for i in range(n)]
    except Exception as e:
        from . import logger
        logger.warning(f"f is not invertible, retrying NTRU lattice generation: {e}")
        return generate_ntru_lattice(n, q)

    B = matrix(ZZ, dim, dim)

    if hasattr(B, 'is_immutable') and B.is_immutable():
        from . import logger
        logger.warning("NTRU matrix is immutable, creating mutable copy")
        B = copy(B)
        B.set_immutable(False)

    for i in range(n):
        B[i, i] = 1
    for i in range(n):
        for j in range(n):
            idx = (j - i) % n
            B[n + i, j] = h[idx]
        B[n + i, n + i] = q

    return B


def apply_lattice_reduction(B, method='LLL', block_size=20):
    """
    Apply lattice reduction to the basis.
    """
    if method.upper() == 'LLL':
        return B.LLL()
    elif method.upper() == 'BKZ':
        return B.BKZ(block_size=block_size)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def gram_schmidt(B):
    """
    Perform Gram-Schmidt orthogonalization on basis B.
    """
    n = B.nrows()
    GS = B.change_ring(QQ)
    mu = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i):
            mu[i][j] = (B[i] * GS[j]) / (GS[j] * GS[j])
            GS[i] -= mu[i][j] * GS[j]

    return GS, mu


def truncate_lattice(B, sigma, center=None, epsilon=1e-10):
    """
    Determine the truncation bounds for efficient lattice sampling.
    """
    GS, _ = gram_schmidt(B)

    n = B.nrows()
    bounds = []
    for i in range(n):
        sigma_i = sigma / GS[i].norm()
        k = np.sqrt(-2 * sigma_i**2 * np.log(epsilon))
        c_i = 0.0 if center is None else center[i]
        lower = int(np.floor(c_i - k))
        upper = int(np.ceil(c_i + k))
        bounds.append((lower, upper))

    return bounds


# ✅ Doctest discovery
import sys
_is_main = __name__ == "__main__"
if not _is_main:
    import doctest
    __test__ = {}
    for name, obj in list(locals().items()):  # ✅ FIXED HERE
        if callable(obj) and hasattr(obj, '__doc__') and '>>>' in obj.__doc__:
            __test__[name] = obj
