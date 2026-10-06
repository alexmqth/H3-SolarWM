# Demo fairness and counting conventions

## Matched inputs

The formal 124-frame W/S/A/D grid keeps the following fixed between original H3 and causal inference:

- initial RGB frame;
- scene prompt;
- action preset and held action sequence;
- seed 13;
- initial video/audio noise;
- resolution and output frame count (124 frames, 24 fps, 5.17 s);
- H3-World checkpoint and patched DiffSynth revision.

The original and causal videos in each side-by-side MP4 therefore share the same starting state and intervention. The formal grid uses the fixed-mix action adapter with the dynamic latent dual-anchor protocol. The later RGB-consistent visual-stability follow-up is a separate adapter/protocol and is labelled separately; it must not be mixed into the formal fixed-mix timing table.

## Step counting

- Original H3: 30 denoiser evaluations over the complete video horizon.
- Causal prototype: 8 denoiser evaluations for each of 8 chunks, or 64 noisy forwards in total, followed by 8 clean KV commits. A clean commit is a history write and is counted separately from the noisy solver steps.
- The phrase “8 steps/chunk” must be used in the presentation. Saying only “8-step H3” would be misleading.

## Timing and memory

The MP4 annotations use one completed recorded run for each action from the formal fixed-mix experiment. They are not a statistical warmup mean. End-to-end timing includes shared setup, conditioning, sampling, VAE preparation/decoding, and MP4 encoding. Sampling-only timing is also included in METRICS.csv.

Peak GPU is torch.cuda.max_memory_allocated in the run metadata. Causal CPU KV is the peak raw-KV cache size. The cache is not included in the original baseline because the original path has no persistent causal cache.

## Quality and continuity

There is no frame-aligned ground-truth future for the generated rollout, so FVD/LPIPS/PSNR and VBench are not reported as if they were supervised quality scores. The meeting table reports:

- mean and p95 adjacent-frame RGB MAD;
- mean RGB MAD at chunk boundary frames 17, 34, 51, 68, 85, 102, and 119;
- signed horizontal Farneback flow;
- visual contact sheets and complete decode checks.

The flow sign is an image-space motion proxy. In this parking-garage scene the original teacher gives positive A and negative D; the causal value is expected to match that geometry if action control is preserved.
