# Stage1 AnyFlow pilot snapshot

Snapshot: 2026-10-08T08:21:13.088919+08:00

Only completed videos are listed. Quality is not automatically accepted.

| Method | E2E s | GPU MiB | Offload reserve GiB | CPU KV MiB | Noisy + commit | Gray MAD | Boundary RGB MAD | Horizontal flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original H3 / A / 30 full-sequence steps | 212.4 | 39923.0 | unrecorded | 0.0 | 30 + 0 | 4.975 | 4.125 | +1.181 |
| Original H3 / D / 30 full-sequence steps | 218.0 | 39923.0 | unrecorded | 0.0 | 30 + 0 | 3.611 | 4.399 | -0.842 |
| anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A / 4 steps/chunk / generated | 228.2 | 38984.2 | 6.0 | 6484.1 | 12 + 3 | 4.072 | 6.069 | -1.199 |
| anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / D / 4 steps/chunk / generated | 220.7 | 38984.2 | 6.0 | 6484.1 | 12 + 3 | 4.048 | 6.485 | -1.242 |
| anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A / 8 steps/chunk / generated | 267.1 | 38984.2 | 6.0 | 6484.1 | 24 + 3 | 3.870 | 5.285 | -1.305 |
| anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / D / 8 steps/chunk / generated | 271.6 | 38984.2 | 6.0 | 6484.1 | 24 + 3 | 3.723 | 4.913 | -1.496 |
| anyflow time=frozen precision=h3_fp32 step00 grid=native / A / 4 steps/chunk / generated | 188.2 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.241 | 6.219 | -1.234 |
| anyflow time=frozen precision=h3_fp32 step00 grid=native / D / 4 steps/chunk / generated | 190.9 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.059 | 6.391 | -1.225 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 4 steps/chunk / generated | 226.7 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.287 | 6.475 | -1.252 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 4 steps/chunk / generated | 214.9 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.064 | 6.268 | -1.213 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 8 steps/chunk / generated | 257.4 | 39069.2 | 6.0 | 6484.1 | 24 + 3 | 3.848 | 5.077 | -1.333 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 8 steps/chunk / generated | 255.2 | 39069.2 | 6.0 | 6484.1 | 24 + 3 | 3.680 | 4.801 | -1.478 |

## A/D numeric checks

- Original H3 / A,D / 30 full-sequence steps: A=+1.1813, D=-0.8421, A-D=2.0234; numeric gate=True; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.1985, D=-1.2421, A-D=0.0436; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.3053, D=-1.4965, A-D=0.1911; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step00 grid=native / A,D / 4 steps/chunk / generated: A=-1.2338, D=-1.2253, A-D=-0.0085; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.2518, D=-1.2127, A-D=-0.0391; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.3333, D=-1.4782, A-D=0.1449; numeric gate=False; visual gate is assessed separately.

## Input fairness against archived original H3

Exact saved-tensor comparison of video/audio noise, prompt, initial image anchor and action row spans; this does not make the different offload configurations a fair speed/memory comparison.

- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 scope=all_qkvo_ffn step16 grid=native / D / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step00 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step00 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 8 steps/chunk / generated: exactly equal = True.

## Fixed validation-noise raw residuals

AnyFlow adaptive scaling can hide large endpoint/general-map residuals in weighted total. Compare these fixed-noise rows before/after training; FM and AnyFlow losses are different objectives.

| Objective | Phase | Action | Sample type | sigma | target sigma | Raw loss | Scale | Weighted loss |
|---|---|---|---|---:|---:|---:|---:|---:|

## Interpretation limits

- MAD measures motion/activity, not video quality.
- Flow signs/separation cannot establish intact geometry or absence of ghosting.
- Runtime is one recorded run with CPU offload and shared hardware, not isolated speedup.
- The archived original run does not record its offload reserve; GPU memory residency is not a matched comparison.
- Initial image/seed/39f references are fixed; a separate seed and 124f/switching remain required.
