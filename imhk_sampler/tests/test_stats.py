"""
Test suite for statistics module.
"""

import numpy as np
import pytest
from sage.all import matrix, vector, RDF

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stats import compute_total_variation_distance


class TestStatistics:
    """Test statistics calculations."""
    
    def test_tv_distance_invalid_sigma(self):
        """Test TV distance with invalid sigma."""
        samples = np.random.randn(100, 2)
        lattice_basis = matrix([[1, 0], [0, 1]])
        
        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_total_variation_distance(samples, 0, lattice_basis)
        
        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_total_variation_distance(samples, -1, lattice_basis)
    
    def test_tv_distance_empty_samples(self):
        """Test TV distance with empty samples."""
        samples = np.array([])
        lattice_basis = matrix([[1, 0], [0, 1]])
        
        result = compute_total_variation_distance(samples, 1.0, lattice_basis)
        assert np.isnan(result)
    
    def test_tv_distance_identity_basis(self):
        """Test TV distance computation with identity basis."""
        # Generate samples around origin
        np.random.seed(42)
        samples = np.random.normal(0, 1, (1000, 2))
        
        lattice_basis = matrix([[1, 0], [0, 1]])
        sigma = 1.0
        
        # Should get a reasonable TV distance
        tv_dist = compute_total_variation_distance(samples, sigma, lattice_basis)
        
        assert 0 <= tv_dist <= 1
        assert not np.isnan(tv_dist)
    
    def test_tv_distance_skewed_basis(self):
        """Test TV distance with skewed basis."""
        samples = np.random.normal(0, 1, (1000, 2))
        
        lattice_basis = matrix([[1, 0.5], [0, 1]])
        sigma = 1.0
        
        tv_dist = compute_total_variation_distance(samples, sigma, lattice_basis)
        
        assert 0 <= tv_dist <= 1
        assert not np.isnan(tv_dist)
    
    def test_tv_distance_with_center(self):
        """Test TV distance with non-zero center."""
        samples = np.random.normal(2, 1, (1000, 2))
        
        lattice_basis = matrix([[1, 0], [0, 1]])
        sigma = 1.0
        center = vector([2.0, 2.0])
        
        tv_dist = compute_total_variation_distance(
            samples, sigma, lattice_basis, center=center
        )
        
        assert 0 <= tv_dist <= 1
        assert not np.isnan(tv_dist)
    
    def test_tv_distance_max_radius(self):
        """Test TV distance with different max_radius values."""
        samples = np.random.normal(0, 1, (1000, 2))
        lattice_basis = matrix([[1, 0], [0, 1]])
        sigma = 1.0
        
        # Test with different radius values
        tv_dist_3 = compute_total_variation_distance(
            samples, sigma, lattice_basis, max_radius=3
        )
        tv_dist_10 = compute_total_variation_distance(
            samples, sigma, lattice_basis, max_radius=10
        )
        
        # Both should be valid
        assert 0 <= tv_dist_3 <= 1
        assert 0 <= tv_dist_10 <= 1
        
        # Both TV distances should be reasonable values
        # Note: We don't enforce that larger radius has smaller TV distance
        # because this depends on the sample quality and distribution
    
    def test_tv_distance_shape_mismatch(self):
        """Test TV distance with shape mismatch."""
        samples = np.random.randn(100, 2)
        lattice_basis = matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3D basis
        
        result = compute_total_variation_distance(samples, 1.0, lattice_basis)
        assert np.isnan(result)  # Should handle gracefully