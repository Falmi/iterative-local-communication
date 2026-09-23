# Local release preparation report

Local Git preparation is complete; public push, release tag and old-repository removal remain pending GitHub CLI authentication.

Current scientific source matches the frozen final manifest. Two gate-related source files differ from Phase 1; their changes add Phase 2 gate mappings. Original snapshots are available. The user-authorized exact 38-file Phase 1 snapshot is now included and verified; the historical-version stop condition is resolved through explicit source mapping.

Original tests: 91 passed. Release tests: 58 passed. Eight CLI help checks, four matrix dry runs, all source imports and config/result parsing passed. All seven manuscript table fragments regenerate byte-for-byte. No long training was run.

Security pattern checks found no flagged candidate content. Datasets and checkpoints are excluded. Exact commands are in README.md. Largest files and package inventory are in docs/release_status.json.

MIT was approved by the author’s instruction to continue. No existing license/institutional mandate was found in the inspected project; institutional ownership is not independently established.

GitHub CLI is absent. After installing the official GitHub CLI, run `gh auth login`, then `gh auth status`. Do not place credentials in files. Neither the replacement repository nor v1.0-paper exists yet. The previous public repository remains unchanged.

Remaining steps: finish independent-clone validation, then publish after CLI authentication.

Proposed Code Availability statement, for use only after successful publication:

“Source code, experiment configurations, seed-specific split metadata, numerical-failure diagnostics, aggregation scripts, and manuscript-level results are publicly available at: https://github.com/Falmi/iterative-local-communication.”

CITATION.cff and README include a citation without invented publication metadata.
