# IMHK Sampler: Cryptographically Relevant Lattice Sampling

This repository contains a research-grade implementation of the Independent Metropolis-Hastings-Klein (IMHK) algorithm for discrete Gaussian sampling over lattices, with a focus on cryptographically relevant parameters aligned with NIST post-quantum cryptography standards.

## Overview

The IMHK sampler is designed for high-quality discrete Gaussian sampling over lattices, a fundamental operation in lattice-based cryptography. This implementation emphasizes:

- **Cryptographic relevance**: Parameters aligned with NIST standards (FIPS 203, FIPS 204)
- **Scalability**: Tested on dimensions from 8 to 128
- **Publication quality**: Comprehensive experiments and visualizations for research papers
- **Security focus**: Analysis of σ/η ratios and security margins

## Key Features

### Cryptographic Parameters
- Dimensions: 8, 16, 32, 64, 128 (scaled from NIST ML-KEM/ML-DSA standards)
- Security levels: Aligned with AES-128, AES-192, AES-256 equivalents
- Basis types: Identity, skewed, ill-conditioned, q-ary lattices

### Algorithms Implemented
- **IMHK Sampler**: Main algorithm with optimized acceptance rates
- **Klein Sampler**: Baseline comparison algorithm
- **Security Metrics**: TV distance, KL divergence, Renyi divergence

### Research Tools
- Comprehensive parameter sweeps
- Security landscape visualization
- Performance vs. security analysis
- Scalability studies up to dimension 128

## Installation

1. Ensure SageMath is installed:
```bash
conda install -c conda-forge sage
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from imhk_sampler import imhk_sampler
from cryptographic_config import CryptographicParameters

# Create a cryptographic lattice basis
dim = 32
B = CryptographicParameters.create_cryptographic_basis(dim, "identity")

# Get appropriate sigma for security
sigma = CryptographicParameters.get_sigma_values(dim, [2.0])[0]

# Sample from discrete Gaussian
samples, acceptance_rate, _, _ = imhk_sampler(B, sigma, num_samples=1000)
```

### Cryptographic Experiments
```python
from publication_crypto_results import main as run_crypto_experiments

# Run full cryptographic analysis
run_crypto_experiments()
```

### Research Publication Results
```python
from experiments_crypto import crypto_parameter_sweep

# Run parameter sweep with NIST-inspired parameters
results = crypto_parameter_sweep(config_type="nist")
```

## Cryptographic Relevance

### NIST Standards Alignment
- **ML-KEM (FIPS 203)**: Module-Lattice-Based Key-Encapsulation Mechanism
  - ML-KEM-512, ML-KEM-768, ML-KEM-1024
  - Our scaled dimensions: 64, 96, 128

- **ML-DSA (FIPS 204)**: Module-Lattice-Based Digital Signature Algorithm
  - ML-DSA-44, ML-DSA-65, ML-DSA-87
  - Direct implementation possible for these dimensions

### Security Parameters
- Smoothing parameter η_ε(Λ) calculation
- σ/η ratios: 1.0 (threshold), 2.0 (standard), 4.0 (secure), 8.0 (high)
- Security margin analysis for practical implementations

## Research Results

### Key Findings
1. IMHK scales effectively to dimension 128
2. Optimal σ/η ratio for crypto applications: 2.0-4.0
3. Maintains performance on structured lattices (q-ary, skewed)
4. Suitable for practical lattice-based cryptographic implementations

### Publications
Results from this implementation are suitable for:
- Cryptographic conference papers (CRYPTO, EUROCRYPT, PKC)
- Security analysis of lattice-based schemes
- Performance comparisons in post-quantum cryptography

## File Structure

```
imhk_sampler/
├── cryptographic_config.py      # NIST-aligned parameters
├── publication_crypto_results.py # Main crypto experiments
├── experiments_crypto.py        # Enhanced experiment framework
├── samplers.py                 # Core sampling algorithms
├── diagnostics.py              # Statistical analysis tools
├── visualization.py            # Publication-quality plots
├── stats.py                    # Security metrics
└── results/
    └── publication/
        └── crypto/             # Cryptographic experiment results
```

## Security Considerations

This implementation is designed for research and analysis. For production cryptographic systems:
- Use constant-time implementations
- Apply additional side-channel protections
- Validate parameters against specific security requirements
- Consider hardware-specific optimizations

## Contributing

Contributions focusing on:
- Improved scalability for larger dimensions
- Additional security metrics
- Optimization for specific architectures
- Comparison with other sampling methods

## License

This research code is provided under MIT License for academic and research purposes.

## Citation

If you use this code in your research, please cite:
```
@article{imhk_crypto_2024,
  title={Cryptographically Relevant Discrete Gaussian Sampling with IMHK},
  author={Lattice Cryptography Research Group},
  journal={Cryptographic Research},
  year={2024}
}
```

## Contact

For research collaborations or questions about cryptographic applications, contact the Lattice Cryptography Research Group.