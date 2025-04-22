# IMHK Sampler for Discrete Gaussian Distributions over Lattices

This repository implements the Independent Metropolis‑Hastings‑Klein (IMHK) sampler for discrete Gaussian distributions over lattices. It is designed for research in small dimensions (2–4 D) to establish empirical baselines for benchmarking quantum speedups.

## Features

- **IMHK and Klein samplers**  
  Core implementations of both the Independent Metropolis‑Hastings‑Klein algorithm and Klein’s algorithm for lattice‐based discrete Gaussian sampling.  
- **Convergence diagnostics**  
  Trace plots, autocorrelation functions, and Effective Sample Size (ESS) calculations to assess sampler performance.  
- **Statistical distance measures**  
  Compute Total Variation distance and Kullback–Leibler (KL) divergence between empirical and ideal distributions.  
- **Visualization tools**  
  2 D/3 D scatter plots, density contours, PCA projections, and marginal histograms for in‐depth sample analysis.  
- **Parameter sweep experiments**  
  Automate large‐scale tests across different dimensions, σ values, lattice bases, and centers to generate comprehensive performance baselines.  
