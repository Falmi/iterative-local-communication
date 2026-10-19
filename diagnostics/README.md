# Numerical failures

[Failure summary](failure_summary.csv) records all six failed runs. Coordinates are one-based; blank values mean unavailable. The C-T5 seed1 entry is an isolated replay of epoch15 validation after global training step5280. C-T3 seed4 failed identically in two unchanged reproductions with valid inputs and labels; the first monitored invalid gradient does not establish the earliest failing backward operator.

Failed seeds were not replaced and were not assigned zero accuracy. Accuracy summaries condition on successful runs. Failure counts do not estimate population failure probabilities.

Main contains manuscript-level summaries only. Detailed original reports and diagnostic JSON remain in the [preservation tag](https://github.com/Falmi/iterative-local-communication/tree/v1.0-full-reproducibility/diagnostics) and [archive branch](https://github.com/Falmi/iterative-local-communication/tree/archive/full-reproducibility/diagnostics). Exact failed model states and large checkpoints were never bundled in these Git snapshots.
