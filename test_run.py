#!/usr/bin/env sage

from classical_sampler.imhk.experiments import run_high_dimensional_test

if __name__ == "__main__":
    result = run_high_dimensional_test(quick_mode=True)
    print("Experiment result:", result)
