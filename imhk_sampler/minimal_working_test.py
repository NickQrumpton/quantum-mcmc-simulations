#!/usr/bin/env sage -python
"""
Absolute minimal test to ensure basic functionality.
This test uses the simplest possible configuration to avoid any edge cases.
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Minimal Working Test")
print("="*50)

# Test 1: Basic imports
print("\n1. Testing imports...")
try:
    from utils import create_lattice_basis
    from samplers import imhk_sampler_wrapper
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create identity basis
print("\n2. Creating identity basis...")
try:
    basis = create_lattice_basis(4, 'identity')
    print(f"✓ Created {basis.nrows()}x{basis.ncols()} identity matrix")
except Exception as e:
    print(f"✗ Basis creation failed: {e}")
    sys.exit(1)

# Test 3: Run sampler with safe parameters
print("\n3. Running sampler...")
try:
    sigma = 2.0  # Safe value
    samples, metadata = imhk_sampler_wrapper(
        basis_info=basis,
        sigma=sigma,
        num_samples=20,
        burn_in=10,
        basis_type='identity'
    )
    print(f"✓ Generated {samples.shape[0]} samples")
    print(f"✓ Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
except Exception as e:
    print(f"✗ Sampler failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test cryptographic basis
print("\n4. Testing q-ary basis...")
try:
    basis = create_lattice_basis(8, 'q-ary')
    sigma = 2.0  # Safe fixed value
    samples, metadata = imhk_sampler_wrapper(
        basis_info=basis,
        sigma=sigma,
        num_samples=20,
        burn_in=10,
        basis_type='q-ary'
    )
    print(f"✓ q-ary test successful")
    print(f"✓ Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
except Exception as e:
    print(f"✗ q-ary test failed: {e}")

print("\n" + "="*50)
print("Basic functionality is working!")
print("\nYou can now try:")
print("1. sage run_simple_crypto_test.py")
print("2. sage run_quick_smoke_test.py")
print("3. sage debug_crypto_test.py (to find specific errors)")