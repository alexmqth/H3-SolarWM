# H3-World x SolarWM: Causalization interview submission

This is the reviewable submission package for the interview task: validate SolarWM-style causal few-step generation in H3-World while checking whether H3-World's original action control survives.

The package contains the causal/KV implementation, small experimental adapters, playable comparison videos, and the key reports. It deliberately excludes the MiniMax-H3 33B base weights, the released H3-World LoRA, datasets, caches, and large latent/conditioning intermediates.

## Result at a glance

- H3-World now runs chunk-wise causal attention, persistent raw KV cache, clean KV commits, generated-history rollout, and action feedback.
- Real 39-frame and 124-frame (about 5.17 seconds) H3 rollouts run end to end. The 124-frame setup uses 8 chunks, 8 solver steps per chunk, 64 noisy denoiser forwards, and 8 clean KV commits.
- The old person transparency/fragmentation problem was traced to an anchor protocol mismatch. Using an RGB-consistent dual anchor (decode generated tail to RGB, then encode it through H3's image branch), together with a tail16 visual QKV adapter, keeps the person and garage structure coherent to the end of the long clips.
- A SolarWM-inspired Stage2-lite chain (student self-rollout, fake-score critic, frozen teacher, DMD surrogate) runs on a shared H3 backbone and has been integrated with the RGB anchor protocol.
- The original H3 A/D action geometry is still not recovered under generated-history causal rollout. The strict gate flow(A)>0, flow(D)<0, and A-D>1.0 is not met. This package does not claim full action preservation.

The defensible final statement is:

> We successfully causalized H3-World with chunk-wise attention and persistent KV caching. RGB-consistent image anchoring and endpoint adaptation recover long-horizon visual stability. However, generated-history causal rollout changes the action-conditioned score geometry, and the strict A/D action gate is not reached. The prototype demonstrates causal feasibility and isolates the remaining action-pathway/rollout-distribution problem; it does not claim full preservation of H3-World action control.

## Documents and layout

- INTERVIEW_ANSWER.md: direct answers to the interview questions.
- EXPERIMENT_REPORT.md: fixed protocols, measurements, and interpretation.
- REPRODUCE.md: environment, patches, external weights, and commands.
- LIMITATIONS.md: claims this package does not make and the next defensible experiments.
- videos/final/: formal 124-frame original-versus-causal comparisons.
- videos/visual_stability/: the RGB-anchor visual-stability breakthrough.
- videos/stage2_lite/: the critic/DMD integration evidence.
- videos/action_geometry/: routing, endpoint, and action-QKV diagnostics.
- breakthrough/: one folder per key breakthrough, with the problem, solution, evidence, and remaining limitation.
- code/: causal, action-routing, and minimal H3 inference code plus DiffSynth patches.
- checkpoints/: small experiment adapters only; no 33B model is included.

## Main videos

| Video | Meaning |
|---|---|
| videos/final/h3world_final_fixed_mix_action_grid_124.mp4 | Formal 124-frame W/S/A/D action grid |
| videos/final/h3world_final_W_original_vs_causal.mp4 | Original 30-step versus causal W |
| videos/final/h3world_final_A_original_vs_causal.mp4 | Original 30-step versus causal A |
| videos/final/h3world_final_D_original_vs_causal.mp4 | Original 30-step versus causal D |
| videos/final/h3world_final_S_original_vs_causal.mp4 | Original 30-step versus causal S |
| videos/visual_stability/stage2_rgb_endpoint_vs_original_W_124.mp4 | RGB-anchor visual-stability repair |
| videos/visual_stability/stage2_rgb_endpoint_visual_stable_AD_124.mp4 | Long A/D visual-stability check; not action-direction proof |
| videos/stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4 | Actual RGB Stage2-lite integration |
| videos/action_geometry/stage2_routing_all_rgb_AD_39.mp4 | Action-row routing upper-bound diagnostic |

All included videos were checked with PyAV as H.264/YUV420P at 24 fps and decode all their frames. The old latent-only cached.mp4 files are intentionally excluded because they used the broken anchor protocol and were not the final playable evidence.

## External weights

Running the code still requires MiniMax-H3 FL2VA base weights, the released H3-World step-10000 LoRA, and the patched DiffSynth checkout. They are intentionally not copied into the submission. See REPRODUCE.md.

## Meeting package

The material for a live interview presentation is collected separately in [`meeting/`](meeting/). It contains timed MP4s, an eight-slide outline, a five-minute script, fairness/counting conventions, and the exact metrics table. Start with [`meeting/README.md`](meeting/README.md).
