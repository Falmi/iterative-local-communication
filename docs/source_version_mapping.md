# Experiment-to-source mapping

| Evidence | Implementation | Verification |
|---|---|---|
| Original Phase 1 A/B/C/D, seeds 1–3 | `historical/phase1/` | All 38 files match the Phase 1 manifest SHA-256 values |
| Phase 2 linear/square-root/cube-root/floor-linear gates | Canonical `src/models/` | Matches Phase 2 model hashes |
| Phase 3 confirmations and Phase 4 residual-scale study | Canonical `src/models/`, original phase-specific training engines | Matches Phase 3/4 model hashes |
| Final CIFAR-100, depth ablations and corruption analysis | Canonical `src/` | Matches the frozen final source manifest |

Two Phase 1 model files differ from the later implementation: `uncertainty.py` and `uncertainty_cellular_classifier.py`. The later version introduces gate mappings and option validation. Both versions are preserved; this release does not assert their byte identity. Model A/B/C files match across phases.

The main five-seed D1 references use the Phase 2 linear gate for seeds 1–3 and Phase 3 for seeds 4–5. The Phase 1 D results remain in the Phase 1 summary and are explicitly associated with the historical snapshot. Public CLI wrappers are packaging adapters, not historical source files. Original per-run provenance remains in the research archive; source hashes and the copy manifest are included here.
