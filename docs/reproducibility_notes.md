# Reproducibility notes

## Scientific source and packaging

All `src/` Python files are byte-identical to the current source pinned by the final-experiment manifest. Some internal protocol modules remain because preserved analysis modules import them; public commands do not require original workspace manifests. Historical training, Phase 4 monitoring, and final-suite monitoring remain separate implementations because merging them would change provenance. The public CLI selects the engine recorded in `configs/index.json`.

Public configs change only dataset/output locations to relative `data` and `runs`. CLI wrappers are new release adapters. Evaluation/analysis entry points use the existing final-suite dataset dispatcher to support CIFAR-100. No model, loss, optimizer, scheduler, seed, batch size or scientific hyperparameter changed. D2/D3/D4 are gate ablations, not primary models. Compatibility configs at the config root support the preserved model unit tests; run matrices use the indexed exact per-run configs.

`docs/source_provenance.json` traces original artifacts and path sanitization. Original source hashes describe original files; release hashes describe the curated bytes. Per-image sample records are omitted from two large phase summaries; all per-seed aggregates remain. Absolute machine paths in records are made relative or replaced with a descriptive local-environment marker. Recorded original hashes are retained as provenance, not represented as hashes of sanitized files.

## Partition verification

Both CIFAR datasets have 50,000 training examples. The unchanged loaders generate `torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()`, use the first 5,000 indices for validation and the remaining 45,000 for training. SHA-256 is computed over `json.dumps(indices).encode()` with default Python JSON formatting. `splits/hashes.json` contains recorded hashes, dataset, seed and sizes. These partitions were regenerated and checked without accessing image files. Seeds 1–5 are supplied for CIFAR-10 and 1–3 for CIFAR-100.

## Outcomes, checkpoint selection and resume

Train 200 epochs; choose the first checkpoint attaining the best validation accuracy, then evaluate test once. Each suite uses the recorded configs. Shared T3 and lambda1 references reuse the same run ID; completed and failed runs are terminal and never rerun by the matrix. Resume only interrupted runs at saved epoch boundaries using the same options (including `--download`). Synthetic `--smoke` uses separate names, CPU, one epoch and four-image batches, and is never manuscript evidence.

Recorded outcomes live in `results/`; new training writes to `runs/` and new aggregation to `outputs/reproduced/`. Generating paper tables from recorded data never launches training and never overwrites archived results. GPU bitwise agreement across environments is not guaranteed. Historical TF32 flags were not fully recorded; diagnostic reports explicitly preserve that limitation.

## Diagnostics and external artifacts

Reports and compact event records are included for C T3 seed4, C T5 seeds1–3 and CIFAR-100 C seeds1–2. The seed4 report compares two unchanged deterministic reproductions, valid inputs/labels and matching sample indices. T5 seed1 includes the isolated epoch15 validation replay. First monitored nonfinite tensors do not identify the earliest failing backward operator.

No checkpoints or datasets are included. The recommended 14-checkpoint archive (A five, D1 five, successful C four) is listed with original hashes and sizes in `checkpoint_archive_plan.json`. C seed4 has no successful final checkpoint and must never be replaced. Diagnostic event checkpoints would need a separate archival deposit to replay the exact captured state. No large artifacts have been uploaded; GitHub Release capacity and institutional/Zenodo deposit policy must be checked when arranging the archive.

## License status

No existing code license or institutional licensing requirement was found in the inspected project. The author authorized proceeding with the proposed MIT code license. This is not a determination of University policy. Dataset licenses and redistribution conditions remain separate; the code license does not cover CIFAR images.

## Historical source versions

The exact 38-file Phase 1 snapshot is included under `historical/phase1/`, with every recorded SHA-256 verified. Later gate-mapping implementations remain in the canonical source. See `source_version_mapping.md` for the phase-to-version mapping. The historical hash discrepancy is resolved by preserving both versions, not by modifying or equating them.
