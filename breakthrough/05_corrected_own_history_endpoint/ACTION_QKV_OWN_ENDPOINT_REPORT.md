# Action QKV with own-history endpoint target

{
  "experiment": "action_qkv_own_endpoint_pair_rgb_39_8step_1update",
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
    "objective": "same-action own-history endpoint x0 MSE weight 0.5 + shared-state full-teacher A/D score delta",
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
    "A": -0.8086314267037731,
    "D": -0.6892747597670869,
    "A_minus_D": -0.11935666693668623
  },
  "latent_delta": {
    "teacher_student_cosine": 0.06553129851818085,
    "student_norm": 129.07029724121094,
    "teacher_norm": 311.4458312988281,
    "chunk_cosines": [
      0.02583104372024536,
      0.06571270525455475,
      0.09513076394796371
    ]
  },
  "resources": {
    "training_seconds": 818.6,
    "gpu_peak_GiB": 39.4,
    "cpu_kv_peak_GiB": 5.28
  },
  "conclusion": "Correct own-history endpoint supervision avoids the mixed-state target bug, but one update still fails the image-space gate: flow(A)=-0.8086, flow(D)=-0.6893, A-D=-0.1194. It is not promoted or expanded."
}
