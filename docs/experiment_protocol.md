# Final manuscript experiment protocol

## Audit and preservation

Model factories, frozen trainer, phase matrix/aggregation code, Phase 1–4 configs,
dataset loaders, deterministic seeding and failure records were inspected. Source
SHA-256, reference checkpoint/result hashes and historical file size/mtime inventory
are captured in final_experiments/manifest.json before execution. No old file is
edited. New code reuses the frozen training loop through isolated dependency bindings.
The C seed-4 failure remains categorical; it has no accuracy or corruption score.

## Models and training

A: CIFAR ResNet-18. B: shared pointwise iterative refinement, historical clean
reference. C: shared depthwise 3x3 local messages with channel normalization,
pointwise transform and residual lambda=1. D1: the exact C modules with detached,
normalized linear predictive-entropy gates. D2/D3/D4 remain ablations.

Training: 200 epochs, batch 128, SGD lr 0.1, momentum 0.9, weight decay 5e-4,
cosine schedule, existing initialization and crop/flip augmentation. Seeded 45k/5k
split; first best validation accuracy chooses a checkpoint; official test is scored
once afterwards. No AMP, clipping, warmup or per-model tuning. Set
CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts; retain existing deterministic
flags. Source/manifest hashes freeze the new implementation before real execution.

CIFAR-100: A/C/D1 T=3 (A has no iterations), seeds 1–3; only normalization
(mean .5071,.4867,.4408; std .2675,.2565,.2761) and class count change. Optional
seeds 4–5 require all nine initial runs successful and an explicit resource decision.
Never replace a failed seed. CIFAR-10 T ablation: C/D1 T=1,2,5 seeds 1–3; reuse T=3.
C-one-shot is mathematically identical to C T=1 and creates no extra training runs.
There are 27 new initial training runs. Numerical failures are terminal outcomes;
infrastructure interruptions remain pending with errors/logs and epoch resume.

## Corruption benchmark

Use only the canonical CIFAR-10-C archive from https://zenodo.org/records/2535967,
linked by https://github.com/hendrycks/robustness. Download verifies the record's
archive checksum. Evaluate all 19 distributed corruption types, including the four
extra types, at all five severities; distinguish the original 15-type subset in
secondary summaries. No corruption training. A/D1 seeds 1–5, C seeds 1,2,3,5.
Means are unnormalized accuracy/error, never called mCE. Equal weight per corruption
and severity within each seed, then summarize independent seed means.

## Metrics and comparisons

Accuracy, NLL, 15-bin ECE, summed multiclass Brier, predictive entropy (natural-log
units), changes across any refinement iteration, per-iteration gates and relative
effective update. Reuse the established uncertainty quartile analysis for D1.
Report successful counts, failures and pending runs. For successful multi-seed
results: mean, sample SD, Student-t 95% CI, median and range. Paired seed differences,
unadjusted paired t-tests/CIs and all seed differences; p>0.05 is not equivalence.
Corruptions/severities are not treated as independent seeds. C conditional means
never become five-seed means. Primary robustness: D1-A, C-A, D1-C. CIFAR-100 same
comparisons. Ablations compare T values within model and one-shot C against A,
C T=3 and D1 T=3. No test-based selection of T or subsequent tuning.

## Compute and output

One consistent PyTorch Conv2d/Linear forward-hook counter at 1x3x32x32 counts dense
MACs including repeated invocations; two arithmetic FLOPs per MAC. Normalization,
activations, pooling, residual adds and entropy/gate operations are excluded and
explicitly listed; partial arithmetic counts are not total model FLOPs. Use existing
matched-protocol GPU validation latency, no CPU latency substitution.
Figures use matplotlib defaults (no manually selected colors), 300-dpi PNG and
vector PDF. Tables are generated from validated records in booktabs format.
The manuscript is revised only after required real evidence is complete and its
source is available. Do not remove the development warning while evidence is pending.
No post-hoc tuning, replacement of failed seeds, or automatic next phase.
