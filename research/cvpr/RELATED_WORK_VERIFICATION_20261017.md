# Related-work checks: 2026-10-17

Judge research note. Discovery used Hugging Face paper search; substantive claims below were checked against primary arXiv pages/HTML. This is a targeted review, not an exhaustive novelty search. No model weights or datasets were downloaded.

## SGF+ — independently verified

- Zihan Su et al., **SGF+: Decoupling Gradient Flows for Autoregressive Video Generation**, [arXiv2610.10429](https://arxiv.org/abs/2610.10429), v1 October7/v2 October8,2026.
- [Primary v2 HTML](https://arxiv.org/html/2610.10429v2), §3.2: reported mean angles104.2 degrees for Attention and106.3 for FFN; the paper also reports negative alignment in512 measured pairs per family. These are the authors' measurements in their setup, not measurements on H3.
- §3.3: separate context-writing and denoising parameters while retaining causal information exchange; writer supervision comes through future prediction.
- §4.1 explicitly describes **two separate LoRA adapters** as one implementation of role separation, although their reported setup uses two full parameter sets. Thus merely using dual LoRA instead of duplicated backbones is explicitly anticipated and cannot be claimed as our novelty.
- Their experimental student is Wan2.1-T2V-1.3B with a frozen14B teacher and a separate fake-score network. Their context-writer/denoiser roles are distinct from our proposed token-provenance partition, but changing the role names does not alone establish novelty.

Implication: our route must demonstrate why action conditioning in a causalized language-controlled model requires a specific intervention, and show behavior and budget-matched gains. Do not label EXP-028's token-gradient cosines a replication of SGF+'s role-gradient result.

## H3-World — independently verified

[H3-World: Turning Language Understanding into World Control](https://arxiv.org/abs/2609.01560), September1,2026. Primary abstract confirms structured character/camera language control, temporal attention routing,8,000 samples/10,000 LoRA updates/0.199% trainable parameters. Our local route audit supplies implementation-specific detail. Direct causal/KV migration remains infrastructure, not an established new contribution.

## GameWAM — neighboring mechanism work

[GameWAM: A World Action Model for Video Games](https://arxiv.org/abs/2608.26200), v2 September10,2026. Primary abstract verified. Jointly generates future observations and keyboard/mouse actions under block-causal conditioning; studies Low-Frequency Action Source Imprinting, where sampled action-source components affect camera motion under fixed conditioning.

It is a policy/world-action task, unlike our externally supplied language-action generation. Nevertheless, it is relevant prior work on mechanism-level source sensitivity and camera/action disentanglement. We cannot claim that source-sensitivity failure analysis in game world models is itself unexplored. Detailed architecture/experiments require full-paper comparison before making narrower claims.

## CAER — still unverified

The quoted title **CAER: Causal Action Effect Reweighting for World Model Training** did not resolve to a primary paper in this bounded search. Exact-title discovery returned unrelated results; HF acronym search returned none; ArcXiv search yielded a client-rendered search shell rather than an identified paper. ArcXiv is an independent reader, not the arXiv primary record. This is an unresolved citation, not proof of nonexistence. Do not cite its claimed method as verified or dismiss possible overlap without an actual paper/identifier.

## Additional primary abstracts verified

- [NoiseGate](https://arxiv.org/abs/2605.07794), May8,2026: learns per-latent timestep schedules as information gating in a joint video/action MoT policy, including changes to latent KV reliability and task-reward optimization. This is a generated-action policy objective, unlike supplied language controls. It is relevant prior art if we later propose noise/time-dependent cache conditioning; changing KV reliability through noise is not an untouched idea.
- [AVControl](https://arxiv.org/abs/2603.24793), March25,2026: separate control-modality LoRAs on parallel conditioning canvases for LTX-2 audio/video generation. Relevant to broad claims about modular control adapters, but not established as a direct baseline for causal language-action history reuse.
- [Modality-Autoregressive World-Action Models](https://arxiv.org/abs/2609.17524), v2 October8,2026: sequentially generates future modalities before actions in robot policies; studies modality choices and training scale. Different task, but modality-specific generation/conditioning is established context.

These facts were checked at abstract level only. Full architecture and experimental comparisons remain to be read before precise novelty claims. Search systems also return irrelevant or potentially later papers; ranking is not evidence of proximity or availability at our research cutoff.

## Consequence for tonight's decisions

1. Finish the bounded EXP-028 measurement; do not enlarge it to obtain stronger negative cosines.
2. If its role differences are weak or already present in Original-visible, do not start dual-adapter training on a gradient-conflict claim.
3. Favor a controlled behavior experiment that can distinguish lawful action-conditioning paths from general quality changes, or a targeted parameter intervention only if both the mechanism evidence and expected behavioral value justify it.
4. Current evidence still does not establish a CVPR method or a first-in-literature claim.
