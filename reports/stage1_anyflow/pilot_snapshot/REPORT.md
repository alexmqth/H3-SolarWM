# Stage1 AnyFlow pilot snapshot

Snapshot: 2026-10-08T07:24:13.120929+08:00

Only completed videos are listed. Quality is not automatically accepted.

| Method | E2E s | GPU MiB | Offload reserve GiB | CPU KV MiB | Noisy + commit | Gray MAD | Boundary RGB MAD | Horizontal flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original H3 / A / 30 full-sequence steps | 212.4 | 39923.0 | unrecorded | 0.0 | 30 + 0 | 4.975 | 4.125 | +1.181 |
| Original H3 / D / 30 full-sequence steps | 218.0 | 39923.0 | unrecorded | 0.0 | 30 + 0 | 3.611 | 4.399 | -0.842 |
| anyflow time=trainable step00 grid=native / A / 4 steps/chunk / generated | 148.9 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.251 | 6.226 | -1.212 |
| anyflow time=trainable step00 grid=native / D / 4 steps/chunk / generated | 155.9 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.086 | 6.604 | -1.196 |
| anyflow time=trainable step04 grid=native / A / 4 steps/chunk / generated | 151.3 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.220 | 6.480 | -1.128 |
| anyflow time=trainable step04 grid=native / D / 4 steps/chunk / generated | 155.8 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.141 | 6.564 | -1.167 |
| anyflow time=trainable step08 grid=native / A / 4 steps/chunk / generated | 151.0 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.289 | 6.416 | -1.289 |
| anyflow time=trainable step08 grid=native / D / 4 steps/chunk / generated | 150.7 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.369 | 6.541 | -1.292 |
| anyflow time=trainable step12 grid=native / A / 4 steps/chunk / generated | 154.0 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.310 | 6.606 | -1.277 |
| anyflow time=trainable step12 grid=native / D / 4 steps/chunk / generated | 162.7 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.308 | 7.118 | -1.313 |
| anyflow time=trainable step16 grid=native / A / 4 steps/chunk / clean | 151.7 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 5.036 | 17.752 | -0.405 |
| anyflow time=trainable step16 grid=native / D / 4 steps/chunk / clean | 153.4 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.226 | 12.307 | -0.919 |
| anyflow time=trainable step16 grid=native / A / 4 steps/chunk / generated | 188.1 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.534 | 7.008 | -1.315 |
| anyflow time=trainable step16 grid=native / D / 4 steps/chunk / generated | 167.1 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 4.230 | 6.778 | -1.224 |
| anyflow time=trainable step16 grid=native / A / 8 steps/chunk / generated | 214.5 | 16247.2 | 20.0 | 6484.1 | 24 + 3 | 3.659 | 5.338 | -1.171 |
| anyflow time=trainable step16 grid=native / D / 8 steps/chunk / generated | 213.5 | 16247.2 | 20.0 | 6484.1 | 24 + 3 | 3.666 | 4.850 | -1.475 |
| fm step16 grid=native / A / 4 steps/chunk / generated | 154.7 | 16437.1 | 20.0 | 6484.1 | 12 + 3 | 3.193 | 3.809 | -1.126 |
| fm step16 grid=native / D / 4 steps/chunk / generated | 148.6 | 16437.1 | 20.0 | 6484.1 | 12 + 3 | 3.191 | 4.101 | -1.360 |
| fm step16 grid=native / A / 8 steps/chunk / generated | 223.4 | 16437.1 | 20.0 | 6484.1 | 24 + 3 | 3.586 | 4.251 | -1.142 |
| fm step16 grid=native / D / 8 steps/chunk / generated | 207.6 | 16437.1 | 20.0 | 6484.1 | 24 + 3 | 3.551 | 4.274 | -1.457 |
| anyflow time=trainable step00 grid=uniform / A / 4 steps/chunk / generated | 149.5 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 5.369 | 10.894 | -2.711 |
| anyflow time=trainable step00 grid=uniform / D / 4 steps/chunk / generated | 150.9 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 5.026 | 9.632 | -2.141 |
| anyflow time=trainable step16 grid=uniform / A / 4 steps/chunk / generated | 151.8 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 5.415 | 11.996 | -2.696 |
| anyflow time=trainable step16 grid=uniform / D / 4 steps/chunk / generated | 148.1 | 16247.2 | 20.0 | 6484.1 | 12 + 3 | 5.094 | 10.025 | -2.253 |
| anyflow time=frozen step16 grid=native / A / 4 steps/chunk / generated | 194.2 | 24617.4 | 20.0 | 6484.1 | 12 + 3 | 4.315 | 6.338 | -1.218 |
| anyflow time=frozen step16 grid=native / D / 4 steps/chunk / generated | 183.8 | 24617.4 | 20.0 | 6484.1 | 12 + 3 | 4.223 | 6.611 | -1.222 |
| anyflow time=frozen step16 grid=uniform / A / 4 steps/chunk / generated | 157.7 | 24617.4 | 20.0 | 6484.1 | 12 + 3 | 5.410 | 11.675 | -2.632 |
| anyflow time=frozen step16 grid=uniform / D / 4 steps/chunk / generated | 155.3 | 24617.4 | 20.0 | 6484.1 | 12 + 3 | 5.139 | 11.052 | -2.310 |
| anyflow time=frozen step16 grid=native / A / 8 steps/chunk / generated | 203.4 | 24617.4 | 20.0 | 6484.1 | 24 + 3 | 3.665 | 5.265 | -1.250 |
| anyflow time=frozen step16 grid=native / D / 8 steps/chunk / generated | 199.5 | 24617.4 | 20.0 | 6484.1 | 24 + 3 | 3.636 | 4.693 | -1.480 |
| anyflow time=frozen step32 grid=native / A / 8 steps/chunk / generated | 235.3 | 39067.7 | 6.0 | 6484.1 | 24 + 3 | 3.655 | 5.146 | -1.332 |
| anyflow time=frozen step32 grid=native / D / 8 steps/chunk / generated | 218.4 | 39067.7 | 6.0 | 6484.1 | 24 + 3 | 3.655 | 4.651 | -1.475 |
| anyflow time=frozen precision=h3_fp32 step00 grid=native / A / 4 steps/chunk / generated | 188.2 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.241 | 6.219 | -1.234 |
| anyflow time=frozen precision=h3_fp32 step00 grid=native / D / 4 steps/chunk / generated | 190.9 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.059 | 6.391 | -1.225 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 4 steps/chunk / generated | 226.7 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.287 | 6.475 | -1.252 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 4 steps/chunk / generated | 214.9 | 39069.2 | 6.0 | 6484.1 | 12 + 3 | 4.064 | 6.268 | -1.213 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / A / 8 steps/chunk / generated | 257.4 | 39069.2 | 6.0 | 6484.1 | 24 + 3 | 3.848 | 5.077 | -1.333 |
| anyflow time=frozen precision=h3_fp32 step16 grid=native / D / 8 steps/chunk / generated | 255.2 | 39069.2 | 6.0 | 6484.1 | 24 + 3 | 3.680 | 4.801 | -1.478 |

## A/D numeric checks

- Original H3 / A,D / 30 full-sequence steps: A=+1.1813, D=-0.8421, A-D=2.0234; numeric gate=True; visual gate is assessed separately.
- anyflow time=trainable step00 grid=native / A,D / 4 steps/chunk / generated: A=-1.2118, D=-1.1956, A-D=-0.0162; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step04 grid=native / A,D / 4 steps/chunk / generated: A=-1.1281, D=-1.1670, A-D=0.0390; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step08 grid=native / A,D / 4 steps/chunk / generated: A=-1.2891, D=-1.2924, A-D=0.0033; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step12 grid=native / A,D / 4 steps/chunk / generated: A=-1.2771, D=-1.3132, A-D=0.0361; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step16 grid=native / A,D / 4 steps/chunk / clean: A=-0.4053, D=-0.9188, A-D=0.5135; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.3153, D=-1.2244, A-D=-0.0909; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.1709, D=-1.4752, A-D=0.3043; numeric gate=False; visual gate is assessed separately.
- fm step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.1260, D=-1.3598, A-D=0.2338; numeric gate=False; visual gate is assessed separately.
- fm step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.1420, D=-1.4568, A-D=0.3149; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step00 grid=uniform / A,D / 4 steps/chunk / generated: A=-2.7111, D=-2.1405, A-D=-0.5705; numeric gate=False; visual gate is assessed separately.
- anyflow time=trainable step16 grid=uniform / A,D / 4 steps/chunk / generated: A=-2.6958, D=-2.2534, A-D=-0.4424; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.2184, D=-1.2215, A-D=0.0031; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen step16 grid=uniform / A,D / 4 steps/chunk / generated: A=-2.6325, D=-2.3101, A-D=-0.3223; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.2504, D=-1.4797, A-D=0.2293; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen step32 grid=native / A,D / 8 steps/chunk / generated: A=-1.3321, D=-1.4745, A-D=0.1424; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step00 grid=native / A,D / 4 steps/chunk / generated: A=-1.2338, D=-1.2253, A-D=-0.0085; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A,D / 4 steps/chunk / generated: A=-1.2518, D=-1.2127, A-D=-0.0391; numeric gate=False; visual gate is assessed separately.
- anyflow time=frozen precision=h3_fp32 step16 grid=native / A,D / 8 steps/chunk / generated: A=-1.3333, D=-1.4782, A-D=0.1449; numeric gate=False; visual gate is assessed separately.

Recorded visual observations: [VISUAL_REVIEW.md](VISUAL_REVIEW.md). Only the videos explicitly covered there have been reviewed.

## Input fairness against archived original H3

Exact saved-tensor comparison of video/audio noise, prompt, initial image anchor and action row spans; this does not make the different offload configurations a fair speed/memory comparison.

- anyflow time=trainable step00 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step00 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step04 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step04 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step08 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step08 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step12 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step12 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=native / A / 4 steps/chunk / clean: exactly equal = True.
- anyflow time=trainable step16 grid=native / D / 4 steps/chunk / clean: exactly equal = True.
- anyflow time=trainable step16 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=native / D / 8 steps/chunk / generated: exactly equal = True.
- fm step16 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- fm step16 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- fm step16 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- fm step16 grid=native / D / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step00 grid=uniform / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step00 grid=uniform / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=uniform / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=trainable step16 grid=uniform / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=native / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=native / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=uniform / A / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=uniform / D / 4 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step16 grid=native / D / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step32 grid=native / A / 8 steps/chunk / generated: exactly equal = True.
- anyflow time=frozen step32 grid=native / D / 8 steps/chunk / generated: exactly equal = True.
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
| anyflow | before | A | diffusion | 0.59218 | 0.59218 | 0.063801 | 1.000000 | 0.103363 |
| anyflow | before | A | diffusion | 0.29619 | 0.29619 | 0.140647 | 1.000000 | 0.189489 |
| anyflow | before | A | endpoint | 0.92971 | 0.00000 | 133.788055 | 0.000764 | 0.037301 |
| anyflow | before | A | flow_map | 0.30200 | 0.20806 | 0.174407 | 0.586088 | 0.139739 |
| anyflow | before | D | diffusion | 0.59218 | 0.59218 | 0.096314 | 1.000000 | 0.156035 |
| anyflow | before | D | diffusion | 0.29619 | 0.29619 | 0.198728 | 1.000000 | 0.267740 |
| anyflow | before | D | endpoint | 0.92971 | 0.00000 | 18.083002 | 0.008158 | 0.053830 |
| anyflow | before | D | flow_map | 0.30200 | 0.20806 | 0.232431 | 0.634659 | 0.201663 |
| anyflow | after | A | diffusion | 0.59218 | 0.59218 | 0.063179 | 1.000000 | 0.102355 |
| anyflow | after | A | diffusion | 0.29619 | 0.29619 | 0.139186 | 1.000000 | 0.187520 |
| anyflow | after | A | endpoint | 0.92971 | 0.00000 | 144.185074 | 0.000702 | 0.036921 |
| anyflow | after | A | flow_map | 0.30200 | 0.20806 | 0.171564 | 0.589733 | 0.138315 |
| anyflow | after | D | diffusion | 0.59218 | 0.59218 | 0.095424 | 1.000000 | 0.154594 |
| anyflow | after | D | diffusion | 0.29619 | 0.29619 | 0.197017 | 1.000000 | 0.265434 |
| anyflow | after | D | endpoint | 0.92971 | 0.00000 | 19.440184 | 0.007522 | 0.053355 |
| anyflow | after | D | flow_map | 0.30200 | 0.20806 | 0.230438 | 0.634504 | 0.199885 |
| fm | before | A | diffusion | 0.59218 | 0.59218 | 0.063900 | 1.000000 | 0.103523 |
| fm | before | A | diffusion | 0.29619 | 0.29619 | 0.140618 | 1.000000 | 0.189450 |
| fm | before | A | diffusion | 0.92971 | 0.92971 | 0.062931 | 1.000000 | 0.022963 |
| fm | before | A | diffusion | 0.30200 | 0.30200 | 0.137354 | 1.000000 | 0.187772 |
| fm | before | D | diffusion | 0.59218 | 0.59218 | 0.096303 | 1.000000 | 0.156018 |
| fm | before | D | diffusion | 0.29619 | 0.29619 | 0.198583 | 1.000000 | 0.267545 |
| fm | before | D | diffusion | 0.92971 | 0.92971 | 0.094984 | 1.000000 | 0.034659 |
| fm | before | D | diffusion | 0.30200 | 0.30200 | 0.194427 | 1.000000 | 0.265795 |
| fm | after | A | diffusion | 0.59218 | 0.59218 | 0.063317 | 1.000000 | 0.102578 |
| fm | after | A | diffusion | 0.29619 | 0.29619 | 0.139569 | 1.000000 | 0.188036 |
| fm | after | A | diffusion | 0.92971 | 0.92971 | 0.059568 | 1.000000 | 0.021736 |
| fm | after | A | diffusion | 0.30200 | 0.30200 | 0.136420 | 1.000000 | 0.186496 |
| fm | after | D | diffusion | 0.59218 | 0.59218 | 0.095589 | 1.000000 | 0.154860 |
| fm | after | D | diffusion | 0.29619 | 0.29619 | 0.197248 | 1.000000 | 0.265745 |
| fm | after | D | diffusion | 0.92971 | 0.92971 | 0.094081 | 1.000000 | 0.034330 |
| fm | after | D | diffusion | 0.30200 | 0.30200 | 0.193140 | 1.000000 | 0.264036 |

## Interpretation limits

- MAD measures motion/activity, not video quality.
- Flow signs/separation cannot establish intact geometry or absence of ghosting.
- Runtime is one recorded run with CPU offload and shared hardware, not isolated speedup.
- The archived original run does not record its offload reserve; GPU memory residency is not a matched comparison.
- Initial image/seed/39f references are fixed; a separate seed and 124f/switching remain required.
