# Visual drift repair: RGB anchor plus generated-history endpoint adaptation

## Question

The first Stage2-lite run used a latent-only previous-chunk anchor while the
tail16 visual adapter had been trained through H3's RGB image-conditioning
branch. Its late chunks showed transparent/fragmented people. This run keeps
the causal/KV protocol fixed and tests whether matching the training anchor
and adding an original-H3 endpoint target stabilizes the generated history.

## Fixed protocol

- 39 RGB frames / 12 latent frames / 3 causal chunks
- 5 latent frames per chunk, history window 5 chunks
- 8 solver steps per chunk, flow shift 2.22
- generated history, persistent raw KV on CPU
- `dynamic_last_frame_rgb_dual`: decode generated prefix to RGB, encode its
  last frame through H3's image branch, and use it as the second dual anchor
- causal action rows and action feedback enabled
- seed 13, fixed initial image/prompt/noise
- fixed-mix action residual; only tail16 causal visual QKV parameters updated
- two online updates, alternating A then D
- teacher replay at every solver sigma plus `0.5 * latent_endpoint_MSE` to the
  original H3 30-step `baseline_latents.pt` and a boundary term

This is an online generated-history visual adaptation. It is deliberately
separate from the critic/DMD action-geometry experiment; it does not claim to
be SolarWM Stage2 SGF/DMD.

## Training

The two updates completed without NaN/OOM. Peak allocated GPU memory was
32,509 MiB on one L40; each detached student/teacher raw-KV cache was about
6.33 GiB on CPU. Update 1 (A) took 553.6 s and update 2 (D) took 522.5 s.

| update | action | sigma replay | endpoint MSE | boundary | grad norm |
|---:|:---:|---:|---:|---:|---:|
| 1 | A | 0.002884 | 0.271850 | 0.002329 | 0.1180 |
| 2 | D | 0.003533 | 0.186136 | 0.002741 | 0.1414 |

## 39-frame generated-history check

Both videos are H.264/YUV420P and decode all 39 frames. The person and the
parking-garage geometry remain visible through frame 38 in both A and D. The
horizontal-flow proxy is still a failure of the separate action gate:

| action | horizontal flow | vertical flow | frames | visual status |
|:---:|---:|---:|---:|:---|
| A | -1.1448 | +0.3306 | 39 | structure stable |
| D | -1.4557 | +0.2567 | 39 | structure stable |

`A-D = 0.3109`; therefore this visual repair must not be described as
recovering A/D action geometry. Its result is that the RGB-consistent anchor
and endpoint target prevent the earlier human decomposition while leaving the
causal action-direction problem unchanged.

## 124-frame long-horizon check

Using the same adapter and protocol, a W rollout completed all 8 chunks:

- 124 frames / 5.17 s
- 64 noisy denoiser forwards + 8 clean KV commits
- 13.19 GiB peak CPU raw-KV cache (history window 5)
- 31,570 MiB peak allocated GPU memory
- sampling time 709.0 s; total after conditioning 752.9 s
- H.264/YUV420P, all 124 frames decode successfully
- horizontal-flow proxy: -0.9391; this is only a motion descriptor

The contact sheet at frames 0, 12, 24, 38, 51, 64, 76, 89, 102, 111 and 123
shows the person and scene remaining coherent through the end of the 5.17 s
clip. This is the first long-horizon visual-stability evidence for the RGB
protocol; it is not evidence that W/S/A/D geometry has been recovered.

The same checkpoint was then run for A and D in parallel on two L40s. Both
videos completed all 124 frames, decode as H.264/YUV420P, and keep the person
and garage geometry through frame 123. Their motion proxies are:

| action | horizontal flow | frame RGB MAD | visual status |
|:---:|---:|---:|:---|
| A | -0.7841 | 2.9972 | stable through frame 123 |
| D | -1.0075 | 2.8282 | stable through frame 123 |

The negative A/D flow signs are consistent with the unresolved action
geometry; they are not a visual-decomposition failure. The A/D 124-frame
comparison is `outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4`.

## Stage2-lite integration check

To make sure the visual repair also survives the actual critic/DMD chain, the
new tail16 visual adapter was plugged into `stage2_lite_dmd.py` with the same
RGB anchor. This round used A/D self-rollouts, target chunks 1 and 2, sigmas
`0.94, 0.79, 0.57, 0.24`, a tail4 fake-score adapter, one shared 33B
backbone, and one student DMD update. It completed in 1111.6 s with 40,320
MiB allocated GPU peak and finite critic/DMD values throughout.

The integrated Stage2-lite A/D check decoded all 39 frames and retained the
person and garage structure through frame 38:

| action | horizontal flow | vertical flow | A-D | visual status |
|:---:|---:|---:|---:|:---|
| A | -1.1419 | +0.3277 | — | stable through frame 38 |
| D | -1.4550 | +0.2577 | 0.3131 | stable through frame 38 |

Adding the critic/DMD update therefore does not reintroduce the old human
decomposition, but it also does not restore A/D action geometry. The
integrated student adapter and logs are in
`stage2_lite_rgb_endpoint_integrated_39_8step_chunks12_sigmas4_1update/`;
the playable A/D comparison is
`outputs/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`.

## Review artifacts

- `contact_sheet.jpg`: A/D at 39 frames
- `eval/AD_flow.json`, `eval/video_stats.json`: numeric diagnostics
- `eval124/W/cached.mp4`: playable 124-frame W rollout
- `eval124/W/contact_sheet.jpg`: long-horizon visual check
- `eval124/A/cached.mp4`, `eval124/D/cached.mp4`: playable 124-frame A/D rollouts
- `eval124/A/contact_sheet.jpg`, `eval124/D/contact_sheet.jpg`: A/D long-horizon checks
- `stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4`: old latent
  Stage2-lite versus the RGB/endpoint result in a four-panel H.264 comparison
- `stage2_rgb_endpoint_vs_original_W_124.mp4`: original 30-step W versus the
  new causal 8-step/chunk W comparison
- `stage2_rgb_endpoint_visual_stable_AD_124.mp4`: causal A/D 124-frame comparison
- `stage2_lite_rgb_endpoint_integrated_AD_39.mp4`: actual critic/DMD-integrated
  RGB-anchor Stage2-lite A/D comparison

The strict action gate remains `flow(A)>0`, `flow(D)<0`, `A-D>1.0`; it remains
intentionally unpassed. The visual drift/decomposition issue is therefore
separated from the unresolved action-geometry issue.
