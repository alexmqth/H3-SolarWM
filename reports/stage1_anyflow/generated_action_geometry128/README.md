# Same-generated-state A/D action geometry diagnostic

**Completed:** [中文结论与归因边界](INTERPRETATION.md) · [Measured results](RESULTS.md) · [Figure](action_geometry.png). All 12 cases completed; no training or demo promotion.

Read-only follow-up to the failed AnyFlow128/136 video experiments. No optimizer,
new architecture, anchor variation, Stage2 update, or demo promotion.

## Controlled intervention

- Checkpoint: `stage1_parallel_resume68_to128/train_128/step_128`.
- Two saved native 8-step/chunk 39-frame trajectories: constant A and constant D.
- Probe chunks 1 and 2 (latent rows 5–9 and 10–11), with 5 and 10 generated latent
  history frames respectively. History is committed with its ORIGINAL actions.
- Hold history, image/audio/noise, anchor, row layout and current state fixed.
  Replace only CURRENT action spans and the matching current residual input.
  Past and future action spans stay unchanged. No branch writes the history KV.
- State is explicitly `(1-sigma)*saved_generated_endpoint + sigma*saved_noise`.
  These interpolants are **not recorded original solver states**.
- Sigma = .9395404663 / .6894410400 / .2407808990; corresponding normal finite
  target = .8694517212 / .5711835327 / 0.

Main result is `student(r=t)` vs Original H3 instantaneous noise-clean velocity.
Finite-map comparisons have different temporal semantics and are labeled
descriptive, not a like-for-like instantaneous geometry test.

## Teacher isolation and comparability

One shared frozen H3 backbone with released hotloaded H3-World action LoRA.
For teacher forwards, disable OUR causal visual adapters, all Stage1 bank
wrappers, AnyFlow target-time conditioner and learned action residual; restore
them afterwards. The released H3-World action LoRA stays active.

Teacher consumes identical clean generated prefix and current noisy tensor,
without future video. Two diagnostic profiles are measured:

1. Matched: same dual RGB anchors, global retiming, fixed prefix times.
2. Native: original single initial anchor and native prefix time policy.

Both use original bidirectional video attention and directed own-action binding.
They are prefix-retake **diagnostics**, not normal original full-horizon
inference. Student reads frozen historical KV whereas teacher recomputes history;
this architectural difference remains. Student's `causal` action prefix policy
also directly exposes past action rows, unlike original own-row binding.

## Checks and controls

- Inspect actual SDPA masks at layers 0/25/49: global frame-to-action mapping,
  historical/current video visibility, own-frame action
  feedback, no prefix reads of historical KV.
- Whole KV content hashes and commit counts before/after all A/D branches;
  per-forward tensor versions/pointers. Model parameter version invariance.
- A/A repeated student and teacher forwards at middle sigma.
- Future-chunk-only action edit at chunk1 as negative control.
- Middle-sigma 2x2 text/residual intervention to separate the two action paths.
- Delta cosine, RMS/norm ratio and per-latent-frame response, plus relative
  action effect against each model's total velocity.

`check_probe.py` exercises the actual tiny H3 implementation, including exact
restoration of unadapted teacher outputs after nonzero visual/bank/time
adaptations, exact student restoration, future-action noninterference, actual
masks and immutable KV. It also verifies real saved action row replacements.
This is implementation validation, not evidence about the pretrained model.

One GPU at a time (GPU6), ≤3 project GPUs remains the session limit. Since another
job entered GPU6 while this probe was prepared, weight reserve is 14 GiB, with
CPU weight offload and a 36000 MiB free-memory preflight. This changes memory
residency, not conditioning, arithmetic dtype, weights or architecture. Do not
use these timings as an inference speed comparison.

The main result must distinguish nonzero action response from correct geometry.
A near-zero cosine is informative only above numerical floor and with a
non-negligible teacher response. Local latent geometry does not establish
image-space strafe direction, video quality, or a Stage2 remedy.
