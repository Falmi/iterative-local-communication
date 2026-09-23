# Reproducibility release report

The curated package is public at https://github.com/Falmi/iterative-local-communication.

The initial code commit is `c63faea8e4611b6f58f5f06b6a736ba9bf12effa`. The paper release tag is `v1.0-paper`; publication metadata may be committed after the initial code snapshot.

There are 302 tracked files, approximately 7.4 MB of curated content. The scientific source and configs are preserved. The exact 38-file Phase 1 snapshot is included with verified hashes and an explicit phase/version mapping.

Validation: 91 original tests passed; 59 release tests passed from an independent Git clone; synthetic CIFAR-100 A training completed; eight CLI help checks and four matrix dry runs passed; all source imports/config/result parsing passed; seven manuscript table fragments regenerate byte-for-byte; 77 config and 110 per-seed summary trace checks passed. No full experiments were rerun.

MIT code license is authorized. Datasets, checkpoints and credentials are excluded. Security checks found no flagged candidate content. Exact reproduction commands and the citation are in README.md.

The recommended 14-checkpoint archive totals 1,272,810,040 bytes and has not been uploaded. Check archive-host limits and dataset/institutional terms separately before depositing large artifacts. No successful C seed4 checkpoint exists. Exact captured failure-state replay requires separately archived model states. Hardware/driver variation and unrecorded historical TF32 flags limit bitwise guarantees.

Code Availability:

“Source code, experiment configurations, seed-specific split metadata, numerical-failure diagnostics, aggregation scripts, and manuscript-level results are publicly available at: https://github.com/Falmi/iterative-local-communication.”

The manuscript is not represented as accepted or published. See CITATION.cff for repository metadata and docs/result_interpretation.md for scientific limitations.
