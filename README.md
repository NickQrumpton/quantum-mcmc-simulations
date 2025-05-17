# Quantum MCMC Simulations

This repository contains classical and quantum algorithm implementations for sampling from discrete Gaussian distributions over lattices, with a focus on testing quantum-enhanced MCMC methods.

## Structure

- `imhk_sampler/`: Main package for Independent Metropolis-Hastings-Klein sampler
- `classical_sampler/`: Additional classical methods (legacy/archive versions)
- `quantum_sampler/`: Quantum-inspired or quantum-enhanced MCMC samplers
- `results/`: Plots and logs from experiments
- `requirements.txt`: Python packages required to run the project

## Getting Started

### Prerequisites

This project requires:
- Python 3.7 or higher
- NumPy, SciPy, Matplotlib, and other scientific packages
- SageMath (for lattice operations)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/quantum-mcmc-simulations.git
   cd quantum-mcmc-simulations
   ```

2. Set up the environment:
   ```
   # Using our helper script to check and setup the environment
   python setup_environment.py --install
   
   # OR install requirements manually
   pip install -r requirements.txt
   ```

3. Install SageMath (required for lattice operations):
   ```
   # Using conda (recommended)
   conda install -c conda-forge sagemath
   
   # OR using pip (limited functionality)
   pip install sagemath
   ```

## Running Tests

Run the minimal test suite to verify your installation:

```
python imhk_sampler/test_minimal.py
```

## Quick Example

```python
from imhk_sampler import imhk_sampler, klein_sampler
from sage.all import matrix, RR

# Create a simple 2D lattice basis
dim = 2
B = matrix.identity(RR, dim)  # Identity basis
sigma = 2.0  # Gaussian parameter
num_samples = 1000

# Generate samples using IMHK
samples, acceptance_rate, _, _ = imhk_sampler(B, sigma, num_samples)
print(f"IMHK Acceptance rate: {acceptance_rate:.4f}")

# Generate samples using Klein's algorithm
klein_samples = [klein_sampler(B, sigma) for _ in range(num_samples)]
```

## Documentation

For more detailed documentation and examples, see the docstrings in the code or run:

```
python imhk_sampler/main.py
```