#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

from sage.all import *
import numpy as np
from samplers import imhk_sampler, klein_sampler

def test_imhk_sampler_signature():
    """Test IMHK sampler with correct parameters."""
    # Create a simple 2x2 identity matrix
    B = identity_matrix(RDF, 2)
    sigma = 1.0
    num_samples = 10
    center = vector(RDF, [0, 0])
    
    try:
        # Call with correct signature
        samples, acceptance_rate, trace, acceptance_trace = imhk_sampler(
            B=B,
            sigma=sigma,
            num_samples=num_samples,
            center=center
        )
        
        print(f"IMHK sampler test passed!")
        print(f"Generated {len(samples)} samples")
        print(f"Acceptance rate: {acceptance_rate:.2%}")
        
    except Exception as e:
        print(f"IMHK sampler error: {e}")

def test_klein_sampler_signature():
    """Test Klein sampler with correct parameters."""
    # Create a simple 2x2 identity matrix
    B = identity_matrix(RDF, 2)
    sigma = 1.0
    center = vector(RDF, [0, 0])
    
    try:
        # Call with correct signature
        sample = klein_sampler(
            B=B,
            sigma=sigma,
            center=center
        )
        
        print(f"Klein sampler test passed!")
        print(f"Generated sample: {sample}")
        
    except Exception as e:
        print(f"Klein sampler error: {e}")

def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    B = identity_matrix(RDF, 2)
    
    # Test with invalid parameter names
    try:
        imhk_sampler(lattice_basis=B, sigma=1.0, size=10)
        print("ERROR: Should have failed with invalid parameter names!")
    except TypeError as e:
        print(f"Correctly caught invalid parameters: {e}")

if __name__ == "__main__":
    print("Testing sampler function signatures...")
    print()
    
    test_imhk_sampler_signature()
    print()
    
    test_klein_sampler_signature()
    print()
    
    test_invalid_parameters()
    print()
    
    print("Signature tests completed!")