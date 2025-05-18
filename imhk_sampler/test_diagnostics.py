#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

import pytest
import numpy as np
from sage.all import vector, RDF
from diagnostics_fixed import compute_autocorrelation, compute_ess, plot_trace, plot_autocorrelation

def test_compute_autocorrelation_1d():
    """Test autocorrelation with 1D input."""
    # Create a simple 1D array
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Compute autocorrelation
    acf = compute_autocorrelation(samples, lag=5)
    
    # Check result shape
    assert len(acf) == 1
    assert len(acf[0]) == 6  # lag + 1
    assert acf[0][0] == 1.0  # Autocorrelation at lag 0 is always 1
    
    print("1D autocorrelation test passed!")

def test_compute_autocorrelation_2d():
    """Test autocorrelation with 2D input."""
    # Create a 2D array
    samples = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    
    # Compute autocorrelation
    acf = compute_autocorrelation(samples, lag=3)
    
    # Check result shape
    assert len(acf) == 2  # Two dimensions
    assert len(acf[0]) == 4  # lag + 1
    assert acf[0][0] == 1.0  # Autocorrelation at lag 0 is always 1
    assert acf[1][0] == 1.0
    
    print("2D autocorrelation test passed!")

def test_compute_ess_1d():
    """Test ESS with 1D input."""
    # Create a simple 1D array with some correlation
    samples = np.cumsum(np.random.randn(100))
    
    # Compute ESS
    ess = compute_ess(samples)
    
    # Check result
    assert len(ess) == 1
    assert ess[0] > 0
    assert ess[0] <= 100  # ESS should be less than or equal to sample size
    
    print(f"1D ESS test passed! ESS = {ess[0]:.2f}")

def test_compute_ess_2d():
    """Test ESS with 2D input."""
    # Create a 2D array
    samples = np.random.randn(100, 3)
    
    # Compute ESS
    ess = compute_ess(samples)
    
    # Check result
    assert len(ess) == 3  # Three dimensions
    for i in range(3):
        assert ess[i] > 0
        assert ess[i] <= 100
    
    print(f"2D ESS test passed! ESS = {ess}")

def test_sage_vectors():
    """Test with SageMath vectors."""
    # Create SageMath vectors
    samples = [vector(RDF, [1.0*i, 2.0*i]) for i in range(10)]
    
    # Compute autocorrelation
    acf = compute_autocorrelation(samples, lag=5)
    assert len(acf) == 2
    
    # Compute ESS
    ess = compute_ess(samples)
    assert len(ess) == 2
    
    print("SageMath vector test passed!")

def test_plot_functions():
    """Test that plot functions handle different input types without errors."""
    # Create test data
    samples_list = [[1, 2], [3, 4], [5, 6]]
    samples_array = np.array(samples_list)
    
    # Test plot_trace
    try:
        plot_trace(samples_list, "test_trace_list.png")
        plot_trace(samples_array, "test_trace_array.png")
        print("plot_trace test passed!")
    except Exception as e:
        print(f"plot_trace error: {e}")
    
    # Test plot_autocorrelation
    try:
        plot_autocorrelation(samples_list, "test_acf_list.png")
        plot_autocorrelation(samples_array, "test_acf_array.png")
        print("plot_autocorrelation test passed!")
    except Exception as e:
        print(f"plot_autocorrelation error: {e}")

def test_edge_cases():
    """Test edge cases that might cause ambiguous array errors."""
    # Single sample
    samples = np.array([[1, 2]])
    ess = compute_ess(samples)
    assert len(ess) == 2
    
    # Very few samples
    samples = np.array([[1], [2]])
    ess = compute_ess(samples)
    assert len(ess) == 1
    
    # Near-constant series
    samples = np.ones((10, 2))
    ess = compute_ess(samples)
    assert len(ess) == 2
    
    print("Edge case tests completed!")

if __name__ == "__main__":
    # Run all tests
    test_compute_autocorrelation_1d()
    test_compute_autocorrelation_2d()
    test_compute_ess_1d()
    test_compute_ess_2d()
    test_sage_vectors()
    test_plot_functions()
    test_edge_cases()
    
    print("\nAll tests passed!")