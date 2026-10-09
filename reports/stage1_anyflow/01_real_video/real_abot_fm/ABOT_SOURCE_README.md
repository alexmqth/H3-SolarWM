---
license: apache-2.0
pretty_name: ABot World Explorer 500h
size_categories:
- 10K<n<100K
tags:
- video
- action-conditioned-video
- world-model
- colmap
- arxiv:2607.19191
dataset_info:
  features:
  - name: sample_id
    dtype: string
  - name: video
    dtype: video
  - name: annotations
    dtype: string
  splits:
  - name: train
    num_examples: 30969
configs:
- config_name: default
  data_files:
  - split: train
    path: metadata.jsonl
---

# ABot World Explorer 500h
[![Studio](https://img.shields.io/badge/Studio-ABot_World_Studio-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMjkgLjMtLjY0LjA4LS41OC0uMjYtLjU3LjI5LS42NC0uMS0uNjQuMzMtLjYyLS4wNS0xLjM1LjYtLjU5LjA0LTEuODcgMS4xMy4xNS4xNy45MS0uMDEgMS4zOS0uNzcgMS4yMi0uMjMuNjUtLjMzIDIuNTgtLjExIDEuMjguNzguNTUuNjkuMzcuOS4wMiAxLjQxLS4zNiAxLjczLS44IDIuMS0xLjkyIDMuNDMtMy4yNSA0LjUuMDIuMy40LjUuNDYuMjQuNTYtLjI4IDMuMTYtNC40LjUzLTEuMzEuODctMS4yOCAxLjY3LTQuNDJ2LTMuMjJsLS40Mi0uMjUtLjI1LS42OS0uNTEtLjY0em0tMS40MyA2LjYyLTMuNjctMi41OS0zLjc2LTEuMjItMi40Ny0uMTktMi42Ny4xOS0zLjcyIDEuMi0zLjY4IDIuNTgtMS4xOCAxLjM4LTEuOTIgMy4wNC0uNzIgMS44Ni0uNTMgMy4wOC0uMDUgMS40NC40NSAzLjA2LjYgMS44NiAxLjg1IDMuMjMgMS44NSAxLjg5LjM5LjE5LjgxLS4xOSAxLjI3LTEuMDYtMi4xOS0yLjM0LS45Mi0xLjM2LS45NS0yLjA4LS41Ny0zLjI3LjMxLTMgLjc0LTIuMjYgMS41OC0yLjU1IDEuMzEtMS40IDMuNDEtMi4xMSAzLjItLjc4aDIuNDVsLjY5LjIzaC44M2wyLjAzLjY5IDIuMyAxLjM5IDIuNSAyLjIzIDEuMTctMS42NS0uMDgtLjY5em0zLjc3IDYuMTQtLjQxLS43NS0uMzYtLjE1LTEuNDMgMi4zNi4xOCAyLjktLjEyIDEuNTgtLjQ0IDEuOTUtMS4xMiAyLjM5LTIuMTMgMi40NC0yIDEuNjEtMi4wMS45NC0xLjk3LjUtMi43MS4xMi0uNTQtLjItLjcyLjA2LTEuMy0uNC0uMTEtLjM2LjUyLS42IDMuNDItMi4zNyAxLjM2LTEuMjIuMDktLjMzLS4yNS0uNDUtLjgzLS40OS01Ljc1IDQuMzgtMy40NiAyLTMuOCAxLjMxLTIuNjMtLjAxLS45My0uNjYtLjY4LS44LS4yNS0xLjIuMTEtMS44MSAxLjU4LTQuMjItLjA2LS4zLS4zLS4xOS0uMzYuMzgtLjE4Ljc1LS42NiAxLjE3LS4zIDEuMzMtLjM0LjU4LS4wMi43OS0uNDUuNDcuMTYuNS0uMTYgMS4xLjIzLjIuMDIuNyAxLjE2IDEuNjkgMS41MS44Ny43LS4wMy41LjI5LjU4LS4yNi43MS4xMS41Ny0uMjkuNzEuMDEgNC4xOS0xLjU5IDEuNzEuNzUgMy4xMy44M2gzLjc4bDMuMTctLjgzIDIuNDUtMS4yIDEuMzMtLjk0IDIuNjQtMi42NyAxLjgzLTMuMjEuMjgtMS4yNi4zNi0uNjR2LS42OWwuNDItLjQxdi00LjI5bC0uMjYtLjI0LjAzLS41NC0uMzYtLjY4em0tOS44OCA0LjE0LS45OC40OS0uOTIuODctLjU1Ljk5LjAyIDEuMTguNzggMS4yNC44Ni43Mi43LjM0IDEuMzMtLjA4LjY3LS4zNCAxLjE3LTEuMjMuMjItLjcyLS4wMS0xLjI1LS4yMi0uNTUtMS4yMS0xLjI1LS42Mi0uMzR6IiBmaWxsPSIjZmZmIi8+PC9zdmc+Cg==)](https://abot-world.amap.com)
[![Playground](https://img.shields.io/badge/Reactor-ABot_World-E9E4C2?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyOCAyMCIgd2lkdGg9IjI4IiBoZWlnaHQ9IjIwIj4KICA8cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZmlsbD0iIzAwMDAwMCIgZD0iTTIzLjg1NjYgMC4zMzQ1NDNDMjYuMjU0OSAwLjMzNDU0MyAyNy4zMTgzIDEuNzQwNDggMjcuMzE4MyAzLjY2MDIzVjkuMzEwNjRDMjcuMzE4MyAxMS4xNzYgMjYuMjgyNyAxMi41ODE5IDIzLjkzODIgMTIuNTgxOUgyMy44ODIyQzIzLjI3MjMgMTIuNTg0IDIyLjkxMTEgMTMuMjYwNiAyMy4yNTIxIDEzLjc2MjFMMjcuMjYzNCAxOS42NjU1SDIyLjM5OTRMMTcuNzU2OSAxMy4wNDg4QzE3LjU1MTUgMTIuNzU2NCAxNy4yMTQ4IDEyLjU4MTkgMTYuODU1NyAxMi41ODE5QzE2LjI4OTEgMTIuNTgxOSAxNS43NTc3IDEzLjA2OTYgMTUuNzU3NyAxMy42NzA5VjE5LjY2NTVIMTEuNTU5NlYxMy42NzA5QzExLjU1OTYgMTMuMDY5NiAxMS4wNjgyIDEyLjU4MTkgMTAuNDYxNSAxMi41ODE5QzEwLjEwMTkgMTIuNTgxOSA5Ljc2NTc3IDEyLjc1NjQgOS41NjAzNSAxMy4wNDg4TDQuOTE3ODMgMTkuNjY1NUgwLjA1NDQyMzJMNC4wNjYyNyAxMy43NjIxQzQuNDA3MjEgMTMuMjYwNiA0LjA0NjUzIDEyLjU4NCAzLjQzNjEzIDEyLjU4MTlIMy4zODAxMUMxLjAzNjE4IDEyLjU4MTkgMCAxMS4xNzYgMCA5LjMxMDExVjMuNjU5N0MwIDEuNzQwNDggMS4wNjI4NSAwLjMzNDU0MyAzLjQ2MTc0IDAuMzM0NTQzSDIzLjg1NjZaTTQuNjg4NCA0LjExOTYzQzQuMzA2OSA0LjExOTYzIDQuMTk4MDYgNC4yNTQ2MiA0LjE5ODA2IDQuNjMzNDRWOC4zMzc0MkM0LjE5ODA2IDguNjg5MDQgNC4zMDY5IDguNzk2ODIgNC42NjE3MiA4Ljc5NjgySDIyLjY1NzdDMjMuMDEyIDguNzk2ODIgMjMuMTIwOCA4LjY4ODUxIDIzLjEyMDggOC4zMzc0MlY0LjYzMzQ0QzIzLjEyMDggNC4yNTUxNSAyMy4wMTIgNC4xMTk2MyAyMi42MzA1IDQuMTE5NjNINC42ODg0WiIvPgo8L3N2Zz4=)](https://reactor.inc/abot-world)
[![Project](https://img.shields.io/badge/🌐_Project-ABot_World-blue)](https://amap-cvlab.github.io/ABot-World/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2607.19191)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/amap-cvlab/ABot-World)  
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/acvlab/ABot-World-0-5B-LF)
[![Space](https://img.shields.io/badge/Space-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/acvlab/abot-world-interactive)
[![Dataset](https://img.shields.io/badge/Dataset_500H-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)
[![Dataset](https://img.shields.io/badge/Dataset_4D-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-4D)
[![Paper](https://img.shields.io/badge/Paper-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/papers/2607.19191)  
[![Model](https://img.shields.io/badge/Model-ModelScope-7061FF?logo=modelscope)](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF)
[![Model](https://img.shields.io/badge/Space-ModelScope-7061FF?logo=modelscope)](https://modelscope.cn/studios/amap_cvlab/abot-world-0)
[![Dataset](https://img.shields.io/badge/Dataset_500H-ModelScope-7061FF?logo=modelscope)](https://modelscope.cn/datasets/amap_cvlab/ABot-World-Explorer-500h)
[![Dataset](https://img.shields.io/badge/Dataset_4D-ModelScope-7061FF?logo=modelscope)](https://modelscope.cn/datasets/amap_cvlab/ABot-World-Explorer-4D)

ABot World Explorer data infrastructure：上游展示海报未在当前工作区找到（原引用 `meta/abot-world-explorer-poster.png`）；不影响实验数据和视频。

ABot World Explorer 500h contains 30,969 action-conditioned video episodes
associated with the data infrastructure described in
[ABot-World-0](https://arxiv.org/abs/2607.19191). Each episode preserves an MP4,
dataset-native keyboard actions, captions, and one COLMAP text sparse model.

## Dataset facts

| Item | Value |
|---|---:|
| Episodes | 30,969 |
| Source objects | 185,814 |
| Semantic splits | None |
| License | Apache-2.0 |

The repository name is an identifier, not an audited duration claim. Exact
duration, FPS, frame count, and alignment statistics are not claimed by this
payload-only publication.

## Layout

```text
meta/abot-world-explorer-poster.png
metadata.jsonl
data/<prefix>/<sample_id>/video.mp4
data/<prefix>/<sample_id>/annotations.tar
LICENSE
README.md
```

`sample_id` is an anonymous HMAC identifier. `annotations.tar` is deterministic,
uncompressed POSIX USTAR containing `action.json`, `caption.json`, and the
complete `sparse/0/{cameras,images,points3D}.txt` COLMAP model. Source keys and
OSS locations are not released.

## Preview

The Hugging Face Dataset Viewer is backed by `metadata.jsonl`. It covers all
30,969 episodes and contains only each anonymous `sample_id`, a typed Video
descriptor with an immutable Hub URI for the existing `video.mp4`, and an
immutable, commit-pinned link to the corresponding `annotations.tar` object.
The typed descriptor lets the Dataset Server provide a playable HTTPS media
source to the Viewer; it does not copy or rewrite the media payload.
It does not copy, rewrite, summarize, or otherwise modify captions, actions, or
media payloads.

## Data file formats

### `video.mp4` and `annotations.tar`

`video.mp4` preserves the source MP4 bytes without release-time transcoding.
`annotations.tar` is an uncompressed POSIX USTAR archive. Treat its five members
as one sample-level annotation package and validate their paths and hashes before
extracting them.

### `action.json`

`action.json` is a UTF-8 JSON object containing sequence metadata and a `frames`
array. The released records use the following structure; the available controls
and dataset-native numeric values may vary by sequence.

| Field | JSON type | Description |
|---|---|---|
| `control_scheme` | string | Name of the source control convention. |
| `original_fps`, `fps` | number | Source and sampled frame rates. |
| `sample_stride` | integer | Sampling stride relative to the source sequence. |
| `start_frame_index`, `end_frame_index`, `total_frames` | integer | Sequence/frame-range metadata. |
| `thresholds` | object | Schema-specific control thresholds; it may be empty. |
| `frames` | array of objects | One ordered control record per sampled frame. |

Each `frames[]` object contains `frame_id` (string), `timestamp` (number), a
`keys` mapping from control names to booleans, and four length-3 numeric vectors:
`delta_translation_cam`, `delta_translation_cam_smooth`, `delta_euler_deg`, and
`delta_euler_deg_smooth`. Rotation deltas are named in degrees; do not infer the
translation or timestamp units, thresholds, or exact cross-modal alignment when
the released schema metadata marks them as unknown. Some source variants add
per-frame translation/rotation threshold and release fields. Readers should
accept additional fields and must not assume a fixed set of keys.

### `caption.json`

`caption.json` is a UTF-8 JSON object with `perspective`, `scene_static`, and
`narrative` strings plus a `dense_temporal` array. `dense_temporal` may be empty;
all current audit canaries contain an empty array, so its item schema is not
claimed here. Consumers must not assume at least one segment. String values are
preserved except that complete-token private source path identifiers are
replaced with `[redacted]`; all other JSON values are preserved, and JSON keys
are never rewritten.

### COLMAP pose files

The three files under `sparse/0/` form one [COLMAP text sparse
model](https://colmap.github.io/format.html#text-format) and must be interpreted
together. Lines beginning with `#` are comments.

- `cameras.txt`: one camera per line as `CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]`.
  The parameter list depends on the [camera
  model](https://colmap.github.io/cameras.html).
- `images.txt`: two lines per image. The first is
  `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`; the second is a repeated list
  of `(X, Y, POINT3D_ID)` observations and may be empty. The pose maps world
  coordinates to camera coordinates using a Hamilton quaternion. COLMAP camera
  axes are +X right, +Y down, +Z forward, and the camera center is `-R^T T`.
- `points3D.txt`: one sparse point per line as
  `POINT3D_ID X Y Z R G B ERROR TRACK[]`; each track item is
  `(IMAGE_ID, POINT2D_IDX)`. A present but empty file is valid and means that the
  model contains no sparse 3D points.

IDs are not guaranteed to be contiguous. See the official specifications for
[`cameras.txt`](https://colmap.github.io/format.html#cameras-txt),
[`images.txt`](https://colmap.github.io/format.html#images-txt), and
[`points3D.txt`](https://colmap.github.io/format.html#points3d-txt).

## Selective download

Use a full commit ID when reproducibility matters. The Preview index provides
the anonymous `sample_id`; the two corresponding payload paths are derived from
that ID without consulting source keys or OSS locations.

```python
from huggingface_hub import snapshot_download

REPO_ID = "acvlab/ABot-World-Explorer-500h"
REVISION = "<full Hugging Face commit ID>"

sample_id = "<sample_id from metadata.jsonl>"
prefix = sample_id[:2]
snapshot = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    revision=REVISION,
    allow_patterns=[
        "LICENSE",
        "README.md",
        "metadata.jsonl",
        f"data/{prefix}/{sample_id}/video.mp4",
        f"data/{prefix}/{sample_id}/annotations.tar",
    ],
)
print(snapshot)
```

Before extracting TAR data, require regular files and the exact member allowlist;
do not call unchecked `extractall()`.

## Intended use and limitations

Intended for research on action-conditioned video prediction, controllable world
models, representation learning, and agent learning. It is not a symbolic
simulator or a guarantee of physical, causal, demographic, or geographic
coverage. Users must assess suitability, bias, safety, and legal obligations for
their downstream use.

## Additional data access
For additional data access or customized dataset requirements, please contact 
the AMAP Data Department directly at
[phys_ai_data@service.alibaba.com](mailto:phys_ai_data@service.alibaba.com)

## Citation

```bibtex
@misc{jiang2026abotworld0infiniteinteractiveworld,
      title={ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU},
      author={Fan Jiang and Zhaoxu Sun and Mengchao Wang and Ziyu Zhu and Chiyu Wang and Yunpeng Zhang and Wenlin Liu and Yun Wang and Xue Zheng and Rui Sun and Junfeng Ni and Hongyu Pan and Zhongxu Sun and Fei Yu and Zengye Ge and Mengmeng Du and Nianfei Fan and Mingchao Sun and Yu Liu and Yongchang and Yanqing Zhu and Jiahang Wang and Ning Ying and Yuze Xuan and Di Yang and Zhicheng Liu and Zhe Gao and Tingbing Xu and Jiacheng Sui and Wenjin Yang and Junnan Lai and Shufeng Liu and Yuan Liu and Zheng Zhou and Yingliang Peng and Dawei Cao and Kaifeng Sheng and Yuxiang Cai and Fei Lu and Mu Xu and Ning Guo},
      year={2026},
      eprint={2607.19191},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.19191},
}
```
