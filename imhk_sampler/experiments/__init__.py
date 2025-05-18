"""
Experiments module for IMHK sampler.

This module provides experimental frameworks and utilities for evaluating
the performance of the IMHK sampler across various parameters and configurations.
"""

# Import key functions from report module
from .report import ExperimentRunner, main

__all__ = ['ExperimentRunner', 'main']