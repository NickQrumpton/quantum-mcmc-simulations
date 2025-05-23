# IMHK Sampler Publication Results

This directory contains the publication-quality results generated by simulations of the Independent Metropolis-Hastings-Klein (IMHK) algorithm for discrete Gaussian sampling over lattices, compared with Klein's algorithm.

## Directory Structure

The results are organized into the following directories:

- **baseline**: Results for basic 2D identity lattice experiments
- **ill_conditioned**: Results showing IMHK's performance on challenging lattice bases
- **parameter_sweep**: Comprehensive comparison across dimensions, sigmas, and basis types
- **convergence**: Analysis of sampling efficiency and effective sample size
- **summary**: Key findings and recommendations for algorithm selection

Each directory contains both raw data (`data/`) and visualizations (`plots/`).

## Key Findings

From our research simulations, we discovered:

1. **Improved Quality**: IMHK sampling achieves better statistical quality than Klein even for well-conditioned lattices, with lower Total Variation distance to the true discrete Gaussian distribution.

2. **Ill-conditioned Performance**: For ill-conditioned lattices, which are challenging for traditional samplers, IMHK shows up to 69.4% improvement in sampling quality over Klein.

3. **Effective Sampling Rate**: When accounting for effective sample size (ESS), IMHK achieves over 5x effective sampling efficiency compared to Klein for higher-dimensional problems.

4. **Acceptance Rate Patterns**: Higher sigma values (σ) generally improve IMHK acceptance rates, with well-conditioned lattices achieving rates above 90%.

## Most Important Files

For a quick overview of the results, check the following files:

1. `summary/research_summary.json`: Contains all key findings and recommendations
2. `summary/algorithm_selection_guide.png`: Visual guide for selecting between IMHK and Klein
3. `ill_conditioned/plots/tv_ratio_comparison.png`: Shows IMHK quality advantage over Klein
4. `convergence/plots/quality_vs_speed.png`: Demonstrates the quality-speed tradeoff
5. `parameter_sweep/plots/tv_ratio_heatmap_ill-conditioned.png`: Heat map of performance by dimension and sigma

## Algorithm Selection Guidelines

- **Use IMHK when**:
  - The lattice basis is ill-conditioned
  - High statistical quality is required
  - Effective sample size is important
  - Working with higher dimensions

- **Use Klein when**:
  - The lattice basis is well-conditioned
  - Raw sampling speed is prioritized
  - Memory constraints are significant
  - Working with very small sigma values

## Data Format

The raw data is saved in two formats:

- **JSON**: Human-readable summary files (e.g., `baseline_summary.json`)
- **Pickle**: Complete Python objects for further analysis (e.g., `sweep_results.pickle`)

## Visualizations

The plots are saved as high-resolution PNG files (300 dpi) suitable for publication. They include:

- Comparison bar charts
- Line graphs showing trends across parameters
- Heat maps for multi-dimensional parameter analysis
- Scatter plots showing quality vs. speed tradeoffs

## Citation

If you use these results in your research, please cite:

```
@article{imhk2025,
  title={Independent Metropolis-Hastings-Klein: A High-Quality Sampler for Discrete Gaussians over Lattices},
  author={Quantum MCMC Research Team},
  journal={Journal of Lattice-Based Cryptography},
  year={2025}
}
```