# Interpreting the results

1. C clean statistics are conditional on successful runs.
2. C seed 4 failed and was not replaced.
3. C T5 failed for all three attempted seeds.
4. CIFAR-100 C failed for two of three seeds.
5. D1 CIFAR-10-C gain has a confidence interval crossing zero.
6. D1 does not outperform A on CIFAR-100.
7. No equivalence conclusion follows from nonsignificant p-values.
8. Reported FLOPs are partial Conv2d/Linear arithmetic counts. Total model FLOPs are unavailable.
9. Failure counts are not population-level failure probabilities.

C T3 exceeds the one-shot C T1 in each of the three paired seeds, but its paired confidence interval includes zero. D1 is an adaptive communication controller, not a universally superior model or a stability guarantee. The 19-corruption CIFAR-10-C mean is unnormalized accuracy, not AlexNet-normalized mCE; the original 15-type subset is separately recorded. Unadjusted comparisons are exploratory. No failures are assigned synthetic accuracy.

The Phase 3 summary retains its historical `incomplete` status (one failed planned seed), whereas final reports use `complete` to mean all planned outcomes were resolved, including failures. These are not conflicting success claims.

Historical table rounding may display C as 95.142 or 95.143 at three decimals owing to binary floating-point rounding of a mean near 95.1425. The original raw seed values and archived table fragments are retained; this is not a changed measurement.
