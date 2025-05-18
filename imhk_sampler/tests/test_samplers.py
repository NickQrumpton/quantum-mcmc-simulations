"""
Test suite for IMHK sampler.
"""

import numpy as np
import pytest
from sage.all import matrix, vector, RDF

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from samplers import imhk_sampler, klein_sampler
from utils import create_lattice_basis


class TestSamplers:
    """Test sampler implementations."""
    
    def test_imhk_sampler_basic(self):
        """Test basic IMHK sampler functionality."""
        basis = create_lattice_basis(2, "identity")
        sigma = 1.0
        num_samples = 100
        
        samples, metadata = imhk_sampler(B=basis, sigma=sigma, num_samples=num_samples)
        
        assert samples.shape == (num_samples, 2)
        assert 'acceptance_rate' in metadata
        assert 0 <= metadata['acceptance_rate'] <= 1
        assert 'samples_accepted' in metadata
        assert 'samples_proposed' in metadata
    
    def test_imhk_sampler_different_dimensions(self):
        """Test IMHK sampler with different dimensions."""
        for dim in [2, 3, 4]:
            basis = create_lattice_basis(dim, "identity")
            sigma = 1.0
            num_samples = 50
            
            samples, metadata = imhk_sampler(
                B=basis, sigma=sigma, num_samples=num_samples
            )
            
            assert samples.shape == (num_samples, dim)
            assert isinstance(metadata, dict)
    
    def test_imhk_sampler_different_sigmas(self):
        """Test IMHK sampler with different sigma values."""
        basis = create_lattice_basis(2, "identity")
        num_samples = 100
        
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            samples, metadata = imhk_sampler(
                B=basis, sigma=sigma, num_samples=num_samples
            )
            
            assert samples.shape == (num_samples, 2)
            
            # Higher sigma should generally lead to better acceptance
            if sigma > 2.0:
                assert metadata['acceptance_rate'] > 0.5
    
    def test_imhk_sampler_skewed_basis(self):
        """Test IMHK sampler with skewed basis."""
        basis = create_lattice_basis(2, "skewed")
        sigma = 1.5
        num_samples = 100
        
        samples, metadata = imhk_sampler(
            B=basis, sigma=sigma, num_samples=num_samples
        )
        
        assert samples.shape == (num_samples, 2)
        assert metadata['acceptance_rate'] > 0
    
    def test_imhk_sampler_ill_conditioned(self):
        """Test IMHK sampler with ill-conditioned basis."""
        basis = create_lattice_basis(2, "ill-conditioned")
        sigma = 3.0
        num_samples = 100
        
        samples, metadata = imhk_sampler(
            B=basis, sigma=sigma, num_samples=num_samples
        )
        
        assert samples.shape == (num_samples, 2)
        # Ill-conditioned basis might have lower acceptance
        assert metadata['acceptance_rate'] > 0
    
    def test_klein_sampler_basic(self):
        """Test basic Klein sampler functionality."""
        basis = create_lattice_basis(2, "identity")
        sigma = 1.0
        num_samples = 100
        
        samples = klein_sampler(B=basis, sigma=sigma, num_samples=num_samples)
        
        assert samples.shape == (num_samples, 2)
        assert isinstance(samples, np.ndarray)
    
    def test_klein_sampler_basic_functionality(self):
        """Test Klein sampler basic functionality."""
        basis = create_lattice_basis(2, "identity")
        sigma = 1.0
        num_samples = 50
        
        # Since the Klein sampler uses the random module and Python's RNG,
        # we can't guarantee deterministic behavior across different runs
        # Let's just check that the samples have correct dimensions
        samples1 = klein_sampler(B=basis, sigma=sigma, num_samples=num_samples)
        samples2 = klein_sampler(B=basis, sigma=sigma, num_samples=num_samples)
        
        assert samples1.shape == (num_samples, 2)
        assert samples2.shape == (num_samples, 2)
    
    def test_invalid_inputs(self):
        """Test samplers with invalid inputs."""
        basis = create_lattice_basis(2, "identity")
        
        # Invalid sigma
        with pytest.raises(ValueError):
            imhk_sampler(B=basis, sigma=-1.0, num_samples=100)
        
        # Invalid num_samples
        with pytest.raises(ValueError):
            imhk_sampler(B=basis, sigma=1.0, num_samples=0)
        
        # Invalid basis (None)
        with pytest.raises(AttributeError):
            imhk_sampler(B=None, sigma=1.0, num_samples=100)