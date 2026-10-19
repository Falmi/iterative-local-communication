# Compact recorded results

CSV values are derived from the validated records preserved at [v1.0-full-reproducibility](https://github.com/Falmi/iterative-local-communication/tree/v1.0-full-reproducibility/results). No measurements were rerun or adjusted.

Accuracy columns are percentages; differences and clean-to-corrupted drops are percentage points. Standard deviations are sample SDs over seeds. Empty fields mean unavailable, including C-T5 accuracy (0/3 successes) and CIFAR-100 C SD (1/3 successes). Latency is milliseconds per sample, synchronized batch128 forward timing; prediction-change frequency is a fraction.

CIFAR-10 C uses four successful seeds of five attempted; iteration T3 summaries use the original three matched seeds. CIFAR-10-C averages all 19 distributed corruption types and five severities equally within seed before averaging seeds. It is unnormalized accuracy, not mCE. The D1−A confidence interval crosses zero. Nonsignificance does not establish equivalence.

Efficiency counts include repeated Conv2d/Linear invocations: one MAC equals two partial arithmetic FLOPs. Normalization, activations, pooling, residual/gate operations, entropy and reductions are excluded. **Total model FLOPs are unavailable.**

`per_seed.csv` retains compact successful outcomes; repeated IDs across suites denote reuse, not independent extra runs. `paired_comparisons.csv` preserves available original paired statistics. Full gate-distribution, uncertainty-quartile, communication-dynamics and per-corruption records remain in the archive. New experiments write to ignored `runs/` and `outputs/`, never these recorded tables.
