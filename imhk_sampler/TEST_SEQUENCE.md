# Test Sequence for IMHK Sampler

Follow these tests in order to verify functionality and identify any issues:

## Step 1: Minimal Test
Start with the absolute minimum to ensure basic functionality:

```bash
sage minimal_working_test.py
```

This tests:
- Basic imports
- Identity lattice creation
- Simple sampling
- Basic cryptographic lattice

## Step 2: Quick Debug
If Step 1 fails, run the debug script:

```bash
sage debug_crypto_test.py
```

This will:
- Test each import individually
- Test each basis type creation
- Identify specific error points

## Step 3: Simple Crypto Test
Once basics work, test all crypto lattices:

```bash
sage run_simple_crypto_test.py
```

This tests:
- All cryptographic basis types
- Safe parameter selection
- Basic TV distance (for small dimensions)

## Step 4: Fixed Crypto Test
Run the comprehensive test with error handling:

```bash
sage run_fixed_crypto_test.py
```

This includes:
- Full error recovery
- Parameter bounds checking
- Detailed logging
- Results saved to JSON

## Step 5: Quick Smoke Test
Test basic experimental framework:

```bash
sage run_quick_smoke_test.py
```

This verifies:
- Experiment framework basics
- Parameter computation
- Basic metrics

## Step 6: Optimized Tests (Optional)
If TV distance is needed:

```bash
sage run_optimized_smoke_test.py --dimensions 8 16 --compute-tv
```

This uses:
- Optimized TV distance computation
- Progress monitoring
- Interrupt handling

## Troubleshooting

### Common Issues and Solutions:

1. **Import Errors**
   - Check that SageMath is properly installed
   - Verify Python path settings
   - Try running from the imhk_sampler directory

2. **Sigma Too Small**
   - The fixed tests include bounds checking
   - Minimum sigma is enforced at 0.1
   - q-ary lattices get special handling

3. **Memory Issues**
   - Start with smaller dimensions
   - Reduce number of samples
   - Skip TV distance computation for large dimensions

4. **Klein Sampler Issues**
   - Usually related to very small sigma
   - Fixed tests include fallback mechanisms
   - Try increasing sigma manually

## Expected Results

When all tests pass, you should see:
- Identity lattice: High acceptance rate (>0.9)
- q-ary lattice: Moderate acceptance rate (0.3-0.8)
- NTRU lattice: Low acceptance rate (<0.1)
- PrimeCyclotomic: Low acceptance rate (<0.1)

## Next Steps

After tests pass:

1. **Run full experiments**:
   ```bash
   sage generate_crypto_publication_results.py
   ```

2. **Generate visualizations**:
   ```bash
   sage run_smoke_test_crypto.py --output-dir final_results
   ```

3. **Check results**:
   - Look at JSON files for detailed metrics
   - Review plots in output directories
   - Check acceptance rates and TV distances

## Files Generated

- `crypto_test_results.json`: Intermediate test results
- `crypto_test_report.json`: Final test report
- `optimized_smoke_test_results.json`: Optimized test results
- Various log files with detailed debugging information