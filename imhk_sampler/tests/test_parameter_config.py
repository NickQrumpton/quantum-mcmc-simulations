"""
Test suite for parameter configuration module.
"""

import numpy as np
import pytest
from sage.all import matrix, vector, sqrt, log, pi

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parameter_config import (
    compute_smoothing_parameter,
    get_experiment_ratios,
    compute_sigma_from_ratio,
    should_skip_experiment,
    get_basis_info,
    generate_experiment_configs
)
from utils import create_lattice_basis


class TestParameterConfig:
    """Test parameter configuration functionality."""
    
    def test_smoothing_parameter_identity(self):
        """Test smoothing parameter for identity lattice."""
        basis = matrix([[1, 0], [0, 1]])
        eta = compute_smoothing_parameter(basis)
        
        assert eta > 0
        assert isinstance(eta, float)
        
        # For identity matrix, smoothing parameter should be predictable
        n = 2
        epsilon = 0.01
        expected = float(sqrt(log(2 * n * (1 + 1/epsilon)) / pi))
        assert np.isclose(eta, expected, rtol=0.1)
    
    def test_smoothing_parameter_skewed(self):
        """Test smoothing parameter for skewed lattice."""
        basis = matrix([[1, 0.5], [0, 1]])
        eta = compute_smoothing_parameter(basis)
        
        assert eta > 0
        assert isinstance(eta, float)
    
    def test_smoothing_parameter_invalid_basis(self):
        """Test smoothing parameter with invalid basis."""
        # Non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            compute_smoothing_parameter(matrix([[1, 0, 0], [0, 1, 0]]))
        
        # Rank-deficient matrix
        with pytest.raises(ValueError, match="must be full rank"):
            compute_smoothing_parameter(matrix([[1, 1], [2, 2]]))
    
    def test_experiment_ratios(self):
        """Test standard experiment ratios."""
        ratios = get_experiment_ratios()
        
        assert isinstance(ratios, list)
        assert len(ratios) == 7
        assert min(ratios) == 0.5
        assert max(ratios) == 8.0
        assert 1.0 in ratios
    
    def test_compute_sigma_from_ratio(self):
        """Test sigma computation from ratio."""
        eta = 1.0
        ratio = 2.0
        
        sigma = compute_sigma_from_ratio(ratio, eta)
        assert sigma == 2.0
        
        # Test minimum sigma constraint
        small_sigma = compute_sigma_from_ratio(0.01, eta)
        assert small_sigma >= 0.1
    
    def test_should_skip_experiment(self):
        """Test experiment skipping logic."""
        # Should skip very small ratios
        skip, reason = should_skip_experiment(0.3)
        assert skip
        assert "below minimum" in reason
        
        # Should not skip normal ratios
        skip, reason = should_skip_experiment(1.0)
        assert not skip
        assert reason == ""
        
        # Should skip with perfect TV distance
        skip, reason = should_skip_experiment(1.0, tv_distance=1e-8)
        assert skip
        assert "perfect mixing" in reason
        
        # Should skip extremely large ratios
        skip, reason = should_skip_experiment(15.0)
        assert skip
        assert "too large" in reason
    
    def test_get_basis_info(self):
        """Test basis information generation."""
        info = get_basis_info("identity", 2)
        
        assert info['type'] == "identity"
        assert info['dimension'] == 2
        assert info['name'] == "identity_2d"
        
        # Test ill-conditioned basis
        info_ill = get_basis_info("ill_conditioned", 3)
        assert 'condition_number' in info_ill
        assert info_ill['condition_number'] == 1000
        
        # Test skewed basis
        info_skew = get_basis_info("skewed", 2)
        assert 'skew_factor' in info_skew
        assert info_skew['skew_factor'] == 0.4
    
    def test_generate_experiment_configs(self):
        """Test experiment configuration generation."""
        dimensions = [2, 4]
        basis_types = ["identity", "skewed"]
        ratios = [0.5, 1.0, 2.0]
        
        configs = generate_experiment_configs(dimensions, basis_types, ratios)
        
        # Should have 2 * 2 * 3 = 12 configurations
        assert len(configs) == 12
        
        # Check all configurations are present
        for config in configs:
            assert config['dimension'] in dimensions
            assert config['basis_type'] in basis_types
            assert config['ratio'] in ratios
        
        # Test empty inputs
        empty_configs = generate_experiment_configs([], [], [])
        assert len(empty_configs) == 0