#!/usr/bin/env python
"""
Unit tests for diagnostics module.
Tests ESS and autocorrelation functions with 1D and 2D arrays.
"""

import numpy as np
import pytest
from sage.all import vector
from diagnostics import compute_autocorrelation, compute_ess


def test_autocorrelation_1d():
    """Test autocorrelation with 1D array."""
    # Generate some correlated data
    n = 100
    data = np.cumsum(np.random.randn(n))  # Random walk
    data = data.reshape(-1, 1)  # Make it 2D with 1 column
    
    acf = compute_autocorrelation(data)
    
    assert isinstance(acf, np.ndarray)
    assert acf.shape == (1, n)
    assert np.abs(acf[0, 0] - 1.0) < 1e-10  # First value should be 1
    

def test_autocorrelation_2d():
    """Test autocorrelation with 2D array."""
    n = 100
    dim = 3
    # Generate some data
    data = np.random.randn(n, dim)
    
    acf = compute_autocorrelation(data)
    
    assert isinstance(acf, np.ndarray)
    assert acf.shape == (dim, n)
    # First value should be close to 1 for each dimension
    for d in range(dim):
        assert np.abs(acf[d, 0] - 1.0) < 1e-10


def test_autocorrelation_sage_vectors():
    """Test autocorrelation with SageMath vectors."""
    n = 50
    dim = 2
    # Create list of sage vectors
    samples = [vector([np.random.randn() for _ in range(dim)]) for _ in range(n)]
    
    acf = compute_autocorrelation(samples)
    
    assert isinstance(acf, np.ndarray)
    assert acf.shape == (dim, n)


def test_ess_1d():
    """Test ESS with 1D array."""
    n = 100
    # Independent data should have ESS close to n
    data = np.random.randn(n).reshape(-1, 1)
    
    ess = compute_ess(data)
    
    assert isinstance(ess, list) or isinstance(ess, np.ndarray)
    assert len(ess) == 1
    # For independent data, ESS should be close to n
    assert ess[0] > n * 0.8  # Allow some tolerance


def test_ess_2d():
    """Test ESS with 2D array."""
    n = 100
    dim = 3
    # Generate independent data
    data = np.random.randn(n, dim)
    
    ess = compute_ess(data)
    
    assert isinstance(ess, list) or isinstance(ess, np.ndarray)
    assert len(ess) == dim
    # For independent data, each dimension should have ESS close to n
    for d in range(dim):
        assert ess[d] > n * 0.8  # Allow some tolerance


def test_ess_sage_vectors():
    """Test ESS with SageMath vectors."""
    n = 50
    dim = 2
    # Create list of sage vectors
    samples = [vector([np.random.randn() for _ in range(dim)]) for _ in range(n)]
    
    ess = compute_ess(samples)
    
    assert isinstance(ess, list) or isinstance(ess, np.ndarray)
    assert len(ess) == dim


def test_empty_arrays():
    """Test handling of empty arrays."""
    # Empty list should raise ValueError
    with pytest.raises(ValueError):
        compute_autocorrelation([])
    
    with pytest.raises(ValueError):
        compute_ess([])
    
    # Empty numpy array should also raise
    with pytest.raises(ValueError):
        compute_autocorrelation(np.array([]))


if __name__ == "__main__":
    # Run tests manually if not using pytest
    test_autocorrelation_1d()
    test_autocorrelation_2d()
    test_autocorrelation_sage_vectors()
    test_ess_1d()
    test_ess_2d()
    test_ess_sage_vectors()
    test_empty_arrays()
    print("All diagnostics tests passed!")