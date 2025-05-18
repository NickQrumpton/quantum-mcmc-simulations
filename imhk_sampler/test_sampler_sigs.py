#!/usr/bin/env sage
"""
Test sampler function signatures.
"""

from sage.all import *
from samplers import imhk_sampler, klein_sampler


def test_imhk_signature():
    """Test IMHK sampler with correct signature."""
    B = matrix(RDF, [[1, 0], [0, 1]])  # 2D identity
    sigma = 1.0
    center = vector(RDF, [0, 0])
    
    # Use correct parameter names
    samples, rate, trace, accepts = imhk_sampler(
        B=B,
        sigma=sigma,
        num_samples=10,
        center=center,
        burn_in=50
    )
    
    assert len(samples) == 10
    assert 0 <= rate <= 1
    print("IMHK sampler test passed!")


def test_klein_signature():
    """Test Klein sampler with correct signature."""
    B = matrix(RDF, [[1, 0], [0, 1]])  # 2D identity
    sigma = 1.0
    center = vector(RDF, [0, 0])
    
    # Use correct parameter names
    sample = klein_sampler(
        B=B,
        sigma=sigma,
        center=center
    )
    
    assert len(sample) == 2
    print("Klein sampler test passed!")


def test_minimal_args():
    """Test samplers with minimal arguments."""
    B = matrix(RDF, [[1, 0], [0, 1]])
    
    # IMHK with minimal args
    samples, _, _, _ = imhk_sampler(B=B, sigma=1.0, num_samples=5)
    assert len(samples) == 5
    
    # Klein with minimal args (center is optional)
    sample = klein_sampler(B=B, sigma=1.0)
    assert len(sample) == 2
    
    print("Minimal arguments test passed!")


if __name__ == "__main__":
    test_imhk_signature()
    test_klein_signature()  
    test_minimal_args()
    print("All sampler signature tests passed!")