"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Generate sample data for tests."""
    np.random.seed(42)
    return {
        'samples_2d': np.random.normal(0, 1, (1000, 2)),
        'samples_3d': np.random.normal(0, 1, (1000, 3)),
        'samples_4d': np.random.normal(0, 1, (1000, 4))
    }


@pytest.fixture
def sage_matrices():
    """Create Sage matrices for tests."""
    from sage.all import matrix
    
    return {
        'identity_2d': matrix([[1, 0], [0, 1]]),
        'identity_3d': matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'skewed_2d': matrix([[1, 0.5], [0, 1]]),
        'ill_conditioned_2d': matrix([[1, 0], [0, 0.01]])
    }


@pytest.fixture
def experiment_params():
    """Standard experiment parameters."""
    return {
        'sigmas': [0.5, 1.0, 2.0, 3.0],
        'ratios': [0.5, 0.75, 1.0, 1.5, 2.0],
        'dimensions': [2, 3, 4],
        'num_samples': 1000,
        'num_chains': 3
    }