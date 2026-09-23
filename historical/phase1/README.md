# Exact Phase 1 implementation

These 38 files are byte-identical to the source snapshot recorded before the original three-seed A/B/C/D experiment matrix. `source_hashes.json` contains the original SHA-256 values. No paths or scientific code inside this snapshot were changed.

This snapshot is historical evidence and an independently runnable implementation, not the canonical later gate-ablation implementation. Run its commands from this directory so Python resolves this version of `src`:

```bash
cd historical/phase1
python -m scripts.train --help
python -m scripts.run_experiment_matrix --help
```

Datasets and outputs are intentionally absent. Use the relative `data/` directory here when reproducing this historical implementation. Do not mix the snapshot and canonical modules in one Python process. The archived Phase 1 D results belong to this implementation; later D1 and D2/D3/D4 use the canonical implementation with explicit gate mappings.
