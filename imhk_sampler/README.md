# IMHK Sampler: Cryptographically Relevant Lattice Sampling

This repository contains a research-grade implementation of the Independent Metropolis-Hastings-Klein (IMHK) algorithm for discrete Gaussian sampling over lattices, with a focus on cryptographically relevant parameters aligned with NIST post-quantum cryptography standards.

## Overview

The IMHK sampler is designed for high-quality discrete Gaussian sampling over lattices, a fundamental operation in lattice-based cryptography. This implementation emphasizes:

- **Cryptographic relevance**: Parameters aligned with NIST standards (FIPS 203, FIPS 204)
- **Scalability**: Tested on dimensions from 8 to 128 with performance optimizations
- **Publication quality**: Comprehensive experiments and visualizations for research papers
- **Security focus**: Analysis of σ/η ratios and security margins
- **Performance**: Optimized TV distance computation with adaptive algorithms

### Recent Updates (2025)

- **Cryptographic Lattice Support**: Added q-ary, NTRU, and Prime Cyclotomic lattice types
- **Performance Optimizations**: Dramatically improved TV distance computation for high dimensions
- **Interrupt Handling**: Graceful termination with partial results
- **Progress Logging**: Real-time monitoring of long-running computations
- **Adaptive Algorithms**: Automatic parameter selection based on dimension

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

## Quick Smoke Test

To verify the installation and test basic functionality:

```bash
sage run_minimal_test.py
```

For optimized smoke test with progress monitoring:

```bash
sage run_optimized_smoke_test.py --dimensions 8 16 --compute-tv
```

## Performance Optimizations

The TV distance computation has been significantly optimized for high dimensions:

### Adaptive Sampling
- Automatic radius selection based on dimension
- Monte Carlo sampling for dimensions > 8
- Early stopping based on convergence criteria

### Progress Monitoring
```python
# Enable progress logging
tv_dist = compute_total_variation_distance(
    samples, sigma, basis,
    progress_interval=5.0,  # Log every 5 seconds
    adaptive_sampling=True  # Use Monte Carlo for high dims
)
```

### Interrupt Handling
- Graceful termination with Ctrl+C
- Partial results saved on interruption
- Diagnostic information available

### Example Usage
```python
from stats import compute_total_variation_distance

# Optimized for high dimensions
tv_distance = compute_total_variation_distance(
    samples, 
    sigma, 
    lattice_basis,
    max_radius=None,          # Auto-computed
    convergence_threshold=1e-4,
    max_points=10000,         # Limit computation
    adaptive_sampling=True,    # Enable for dim > 8
    progress_interval=5.0      # Progress logs
)
```

See `OPTIMIZATION_DOCUMENTATION.md` for detailed performance guidelines.

This quick test:
- Tests a 2D example with visualization
- Runs diagnostics for all basis types
- Computes all metrics (TV distance, acceptance rate, ESS)
- Takes approximately 10 seconds
- Creates plots in `results/smoke_test/`

Expected output:
```
=== IMHK Sampler Smoke Test Results ===
2D Example: ✓
Identity Basis: ✓
Skewed Basis: ✓
Ill-conditioned Basis: ✓
All tests passed! Time: 9.45 seconds
```

## Full Publication Run

To generate full publication-quality results with comprehensive TV distance comparisons:

```bash
sage run_full_tv_results.py
```

### NEW: Publication-Quality Experiments

For the comprehensive refactored experiments with adaptive parameter configuration:

```bash
python run_publication_experiments.py
```

This new experiment framework provides:
- Error handling focused solely on TV distance metrics
- Adaptive parameter configuration using smoothing parameter η_ε(Λ)
- Comprehensive experiments across basis types and dimensions
- Performance logging throughout execution
- Professional visualization with error bars
- Automated report generation with key findings

Results include:
- CSV and JSON data files with all metrics
- Publication-quality plots (PNG, 300 DPI)
- Comprehensive experiment report
- Optimal parameter recommendations

This comprehensive run:
- Tests dimensions: 2, 4, 8, 16, 32, 64
- Uses basis types: identity, skewed, ill-conditioned
- Tests σ/η ratios from 0.5 to 8.0
- Generates 10,000 samples per experiment
- Creates publication-quality plots
- Saves detailed results and summary statistics

Estimated runtime: 4-8 hours (depends on hardware)

Results saved to:
- `results/publication_tv/tv_distance_comparison.json` - Numerical results
- `results/publication_tv/summary.json` - Experiment summary and key findings
- `results/publication_tv/plots/` - Publication-quality visualizations

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

## Testing

### Run Unit Tests

To run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=imhk_sampler --cov-report=html

# Run specific test modules
pytest tests/test_stats.py -v
pytest tests/test_parameter_config.py -v
pytest tests/test_samplers.py -v

# Run without slow tests
pytest tests/ -m "not slow"
```

### CI/CD

The project includes GitHub Actions workflow for:
- Running tests on Python 3.8, 3.9, 3.10
- Code formatting checks with Black
- Linting with flake8
- Coverage reporting to Codecov
- Running example scripts

## Documentation Style

All code follows:
- PEP8 formatting (enforced with Black)
- NumPy-style docstrings
- Type hints for all function signatures
- Comprehensive logging throughout
- Error handling with informative messages

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