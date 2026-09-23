# Numerical failure report: two CUDA reproductions analyzed

Both unchanged diagnostic runs detected the same event at **epoch 1, batch 73, global step 73 (all 1-based), learning rate 0.1**. The full recorded failure and preceding-step records are identical, including sample indices. Environment and diagnostic runner hashes match. Reproducibility is deterministic at recorded resolution, not a proof about all CUDA executions.

## Primary classification

**B — non-finite backward gradients following severe numerical growth.** All monitored forward outputs and the loss at step 73 are finite. Loss is 4.68163365e18 (step 72: 5.68563455e18; first batch: 2.46871233). Backbone and communication gradients are non-finite after backward; classifier gradients remain finite with norm 4.1836148e9. The first invalid entry in parameter-list order is `backbone.0.weight`; the first in the communication group is `communication_blocks.0.neighbourhood_aggregation.weight`.

This does not locate the earliest failing backward operation. No backward intermediate hooks were recorded, so the named backbone parameter must not be treated as proof that the backbone originated the NaN. Internal normalization arithmetic may differ from the float64 diagnostic variance reductions; those reductions do not prove the float32 kernel intermediates remained finite.

## Optimizer and communication evidence

Step 72 gradients, parameters after its optimizer update, and momentum buffers were finite. Parameter norms nevertheless reached backbone 5.86318e9, communication 2.69497e11, classifier 1.01644e10. Step 73 stopped before optimizer.step; its invalid gradients were not applied. Thus the first monitored non-finite event is not an optimizer parameter update with finite gradients.

At step 73, communication state/message norms are:

| Iteration | State norm before | Message norm | Message/state ratio |
|---|---:|---:|---:|
| 1 | 1.54956e10 | 6.67957e10 | 4.31063 |
| 2 | 6.60100e10 | 6.67957e10 | 1.01190 |
| 3 | 1.31901e11 | 6.67957e10 | 0.506408 |

Final state norm is 1.98398e11. These observations show already severe numerical growth and further residual accumulation, but do not establish which earlier update initiated divergence or prove communication is the unique cause. Successful C seeds have not been instrumented at the matching stage.

## Data and exact-batch replay

Both runs use identical sample indices. Inputs are finite, within expected normalized ranges (observed min -1.98947, max 2.12649), and labels are int64 in [0,9]. Replaying the exact augmented batch from the pre-batch state gives finite train-mode loss 4.68163e18 and finite eval-mode loss 1.31746e26. These are numerically extreme, not normal behavior. Replay checked forward/loss only, not backward. Matching sample indices across deterministic runs is not evidence of data corruption. No malformed sample was demonstrated.

## Environment

RTX A6000; Python 3.13.5; PyTorch 2.9.1+cu130; float32 inputs/model; AMP off; deterministic algorithms and cuDNN deterministic on; cuDNN benchmark off; matmul TF32 off; cuDNN TF32 on. Historical TF32 flags were not recorded, so historical equality cannot be certified. The NVML warning does not explain the observed failure.

## Scientific impact and next decision

1. Both diagnostic runs failed at the identical location stated above.
2. The first monitored non-finite quantity is a parameter gradient after backward; the earliest backward operation remains unlocalized.
3. The same batch and recorded numerical statistics occur in both runs.
4. Evidence supports deterministic numerical divergence culminating in invalid gradients, rather than a differing-location runtime failure.
5. No general implementation defect or corrupted sample has been demonstrated. These observations do not invalidate completed C or Phase 1/2 results, but also do not establish that only seed 4 can fail.
6. There is no evidence that another unchanged standard run will now complete. The diagnostic stop precedes the original trainer's loss-only guard; the later original non-finite loss was not directly reproduced because applying invalid gradients was deliberately prevented.
7. Recommendation: **keep seed 4 recorded as a failed run under the current protocol**. Do not launch another standard run, replace the seed, or declare Phase 3 complete. Whether a protocol-level intervention is warranted remains a scientific decision; if adopted, declare it separately and rerun comparisons consistently.

No scientific code, config, successful result, or failed run status was modified. Prior blocked reports were preserved with `.blocked_backup` suffixes. Full data remain in diagnostic_run1.json and diagnostic_run2.json; this report supersedes the blocked-status conclusion while retaining the measurement limits above.
