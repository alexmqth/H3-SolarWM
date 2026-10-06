# Action QKV + original-H3 endpoint target

{
  "experiment": "action_qkv_latent_endpoint_pair_rgb_39_8step_1update",
  "protocol": {
    "frames": 39,
    "latent_frames": 12,
    "chunks": 3,
    "chunk_frames": 5,
    "solver_steps": 8,
    "flow_shift": 2.22,
    "anchor": "RGB dual",
    "history": "generated",
    "kv": "persistent CPU raw KV",
    "seed": 13,
    "visual_adapter": "visual_online_rgb_tail16_endpoint_ad2",
    "student": "zero-init tail4 action QKV",
    "objective": "paired full-teacher A/D score delta + 0.5 original H3 endpoint latent MSE",
    "target_chunks": [
      1,
      2
    ],
    "sigmas": [
      0.94,
      0.79,
      0.57,
      0.24
    ]
  },
  "flow": {
    "A": -0.80412807315588,
    "D": -0.7047332937976247,
    "A_minus_D": -0.09939477935825525
  },
  "latent_delta": {
    "teacher_student_cosine": 0.05633892863988876,
    "rows": [
      {
        "kind": "teacher",
        "delta_norm": 311.4458312988281,
        "delta_rms": 0.4646466374397278
      },
      {
        "kind": "student",
        "delta_norm": 120.48351287841797,
        "delta_rms": 0.1797650307416916
      }
    ],
    "chunk_cosines": [
      -0.013728431425988674,
      0.0566425696015358,
      0.08859824389219284
    ]
  },
  "visual_note": "RGB anchor keeps this 39-frame clip decodable; no new 124-frame rollout was promoted.",
  "conclusion": "This naive mixed endpoint target slightly rotates the latent action delta (cosine 0.016 to 0.056) but does not recover image-space action control. Because the paired branch uses an A-generated counterfactual state while the D endpoint comes from an independent D rollout, this is a state-mismatch negative diagnostic rather than a general endpoint-supervision conclusion. The corrected own-history protocol is recorded separately."
}
