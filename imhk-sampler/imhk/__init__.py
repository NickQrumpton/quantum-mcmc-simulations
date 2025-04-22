# Expose key functions for easy imports
from .samplers import imhk_sampler, klein_sampler, discrete_gaussian_sampler_1d
from .utils import discrete_gaussian_pdf, precompute_discrete_gaussian_probabilities
from .diagnostics import compute_autocorrelation, compute_ess, plot_trace, plot_autocorrelation, plot_acceptance_trace
from .visualization import plot_2d_samples, plot_3d_samples, plot_2d_projections, plot_pca_projection
from .stats import compute_total_variation_distance, compute_kl_divergence
from .experiments import run_experiment, parameter_sweep, plot_parameter_sweep_results, compare_convergence_times