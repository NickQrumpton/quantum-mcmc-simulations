"""
Cryptographic configuration for IMHK sampler based on NIST standards.

This module defines cryptographically relevant parameters for lattice-based
sampling experiments, aligned with NIST post-quantum cryptography standards
(FIPS 203, FIPS 204) and current research needs.
"""

from typing import List, Dict, Tuple
import numpy as np
from sage.all import RR, matrix
import logging

logger = logging.getLogger("imhk_crypto_config")

class CryptographicParameters:
    """
    Configuration class for cryptographically relevant lattice parameters.
    
    Based on NIST standards:
    - ML-KEM (FIPS 203): Module-Lattice-Based Key-Encapsulation Mechanism
    - ML-DSA (FIPS 204): Module-Lattice-Based Digital Signature Algorithm
    """
    
    # Cryptographically relevant dimensions
    # These are practical dimensions attainable by IMHK sampler
    # while maintaining cryptographic relevance
    CRYPTO_DIMENSIONS = {
        "small": 8,      # Minimal cryptographic relevance
        "medium": 16,    # Common in learning with errors 
        "large": 32,     # Standard for many lattice schemes
        "xlarge": 64,    # High security applications
        "max": 128,      # Maximum practical dimension for IMHK
    }
    
    # NIST-inspired dimensions (scaled down for feasibility)
    NIST_INSPIRED_DIMENSIONS = {
        "ml-kem-512-scaled": 64,   # Scaled from 512
        "ml-kem-768-scaled": 96,   # Scaled from 768
        "ml-kem-1024-scaled": 128, # Scaled from 1024
        "ml-dsa-44-scaled": 44,    # Direct from ML-DSA-44
        "ml-dsa-65-scaled": 65,    # Direct from ML-DSA-65
        "ml-dsa-87-scaled": 87,    # Direct from ML-DSA-87
    }
    
    # Research-focused dimensions for publication
    RESEARCH_DIMENSIONS = [8, 16, 32, 64]
    
    # Gaussian parameter configurations
    # σ/η ratios for cryptographic security
    SIGMA_ETA_RATIOS = {
        "minimal": 1.0,    # Threshold for security
        "standard": 2.0,   # Common in applications
        "secure": 4.0,     # Conservative choice
        "high": 8.0,       # High security margin
    }
    
    @classmethod
    def get_sigma_values(cls, dimension: int, 
                        ratios: List[float] = None) -> List[float]:
        """
        Calculate appropriate sigma values for a given dimension.
        
        Args:
            dimension: Lattice dimension
            ratios: List of σ/η ratios (default: use standard ratios)
            
        Returns:
            List of sigma values
        """
        if ratios is None:
            ratios = list(cls.SIGMA_ETA_RATIOS.values())
        
        # Calculate smoothing parameter
        epsilon = 2**(-dimension)  # Standard choice
        eta = np.sqrt(np.log(2 * dimension / epsilon) / np.pi)
        
        # Calculate sigma values from ratios
        sigmas = [ratio * eta for ratio in ratios]
        
        # Round to reasonable precision
        sigmas = [round(sigma, 2) for sigma in sigmas]
        
        return sigmas
    
    @classmethod
    def get_experiment_config(cls, 
                             config_type: str = "research") -> Dict[str, any]:
        """
        Get experiment configuration for different scenarios.
        
        Args:
            config_type: Type of configuration ("research", "crypto", "nist")
            
        Returns:
            Dictionary with experimental parameters
        """
        configs = {
            "research": {
                "dimensions": cls.RESEARCH_DIMENSIONS,
                "sigmas": lambda dim: cls.get_sigma_values(dim),
                "basis_types": ["identity", "skewed", "ill-conditioned"],
                "num_samples": 2000,
                "burn_in": 1000,
                "description": "Research publication configuration"
            },
            "crypto": {
                "dimensions": list(cls.CRYPTO_DIMENSIONS.values()),
                "sigmas": lambda dim: cls.get_sigma_values(dim),
                "basis_types": ["identity", "skewed"],
                "num_samples": 5000,
                "burn_in": 2000,
                "description": "Cryptographically relevant configuration"
            },
            "nist": {
                "dimensions": [
                    cls.NIST_INSPIRED_DIMENSIONS["ml-dsa-44-scaled"],
                    cls.NIST_INSPIRED_DIMENSIONS["ml-dsa-65-scaled"],
                    cls.NIST_INSPIRED_DIMENSIONS["ml-kem-512-scaled"],
                ],
                "sigmas": lambda dim: cls.get_sigma_values(dim, [2.0, 4.0, 8.0]),
                "basis_types": ["identity", "skewed"],
                "num_samples": 10000,
                "burn_in": 5000,
                "description": "NIST-inspired scaled configuration"
            }
        }
        
        if config_type not in configs:
            raise ValueError(f"Unknown configuration type: {config_type}")
            
        return configs[config_type]
    
    @classmethod
    def create_cryptographic_basis(cls, dim: int, 
                                  basis_type: str = "identity") -> matrix:
        """
        Create a cryptographically relevant lattice basis.
        
        Args:
            dim: Dimension of the lattice
            basis_type: Type of basis to create
            
        Returns:
            Lattice basis matrix
        """
        if basis_type == "identity":
            B = matrix.identity(RR, dim)
        elif basis_type == "skewed":
            # Create a basis similar to those in cryptographic applications
            B = matrix.identity(RR, dim)
            # Add controlled skew similar to q-ary lattices
            for i in range(min(dim-1, 4)):  # Limit skew to avoid numerical issues
                B[i, i+1] = RR(2**(i+1))
        elif basis_type == "ill-conditioned":
            # Simulate a basis with properties of hard lattices
            B = matrix.identity(RR, dim)
            # Create exponentially growing basis vectors
            for i in range(dim):
                B[i, i] = RR(2**(i/2))
        elif basis_type == "q-ary":
            # q-ary lattice structure common in LWE
            q = 2**16  # Common modulus in crypto
            B = matrix.identity(RR, dim)
            B[0, 0] = q
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
        
        return B
    
    @classmethod
    def get_security_parameters(cls, dimension: int) -> Dict[str, any]:
        """
        Get security-relevant parameters for a given dimension.
        
        Args:
            dimension: Lattice dimension
            
        Returns:
            Dictionary with security parameters
        """
        # Approximate security levels
        if dimension <= 32:
            security_level = "Low (Research/Testing)"
            bit_security = dimension * 2
        elif dimension <= 64:
            security_level = "Medium"
            bit_security = dimension * 1.5
        elif dimension <= 128:
            security_level = "High"
            bit_security = min(dimension, 128)
        else:
            security_level = "Very High"
            bit_security = 128
        
        return {
            "dimension": dimension,
            "security_level": security_level,
            "estimated_bit_security": int(bit_security),
            "smoothing_parameter": np.sqrt(np.log(2 * dimension / 2**(-dimension)) / np.pi),
            "recommended_sigmas": cls.get_sigma_values(dimension),
            "description": f"{dimension}-dimensional lattice configuration"
        }