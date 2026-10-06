# 复现说明

本包是提交和审阅包，不带 33B 权重。复现需要 CUDA 机器、H3-World 权重和 patched DiffSynth。项目原始工作目录可参考本包的 code/ 和 PROJECT_PROGRESS.md。

## 1. 环境

原实验使用 Python 3.10、CUDA 12.8、单张或多张 48 GiB L40。依赖版本见 requirements.txt：

    cd submission
    conda create -n h3world python=3.10 -y
    conda activate h3world
    pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    source env.sh
    export DIFFSYNTH_SKIP_DOWNLOAD=True

如果使用已有环境 `/home/lpeng/miniconda3/envs/h3world/bin/python`，只需激活该环境并执行 `source env.sh`。不要在没有明确准备好存储空间时运行自动 ModelScope/Hugging Face 下载。

## 2. 外部代码和权重

在 submission/ 下准备 patched DiffSynth checkout：

    git clone https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3-v2
    git -C DiffSynth-Studio-h3-v2 checkout "$(cat code/diffsynth_base_commit.txt)"
    git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_h3_action.patch

如果需要 causal patch，再应用：

    git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_causal.patch

准备以下外部文件（不放入提交包）：

    DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/   # MiniMax-H3 FL2VA base
    checkpoints/H3-World/step-10000.safetensors       # H3-World released LoRA

`infer.py` 会检查加载的 DiffSynth 是否来自 `-v2` checkout，并检查 directed attention mask；如果检查失败会拒绝运行，避免不匹配的 attention patch 产生无效结果。

## 3. 39 帧 causal smoke

从 submission 根目录运行，benchmark.py 会把 code/abot 和 code 加入 Python path：

    CUDA_VISIBLE_DEVICES=0 python code/causal/benchmark.py \
      --modes cached \
      --steps 8 \
      --flow-shift 2.22 \
      --num-frames 39 \
      --chunk-frames 5 \
      --history-chunks 5 \
      --action-preset A \
      --cache-device cpu \
      --anchor-mode dynamic_last_frame_rgb_dual \
      --seed 13 \
      --causal-adapter checkpoints/visual_rgb_tail16/causal_adapter.pt \
      --causal-action-adapter checkpoints/visual_rgb_tail16/action_adapter.pt \
      --action-prefix-mode causal \
      --action-feedback \
      --out-dir outputs/reproduce_A_39

将 `--action-preset A` 换成 `D` 可以生成对应 D 片段。原始 baseline 可以用 `--modes baseline --baseline-steps 30` 运行。实际生成还需要脚本默认位置可找到 initial frame、scene prompt 和 checkpoint；具体路径按 code/abot/infer.py 中的 H3-World 目录配置填写。

## 4. Stage2-lite diagnostic

Stage2-lite 的实现是 code/causal/stage2_lite_dmd.py。它支持 `--latent-target-weight` 和 `--own-latent-target-weight`；后者使用同动作 own-history endpoint，避免把 A state 和独立 D state 混用。完整参数以：

    python code/causal/stage2_lite_dmd.py --help

为准。推荐从 39 帧、1 update、chunk 1/2、CPU raw KV 开始。该诊断很慢，单次实验约十几分钟到更久；不应一开始扩到 124 帧或 158 帧。

## 5. 只验证视频可播放性

不需要加载模型即可检查本包视频：

    python - <<'PY'
    from pathlib import Path
    import av
    for p in Path('videos').rglob('*.mp4'):
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            n = sum(1 for _ in c.decode(video=0))
            print(p, 'frames=', n, 'codec=', s.codec_context.name,
                  'pix_fmt=', s.codec_context.format.name if s.codec_context.format else None,
                  'fps=', float(s.average_rate) if s.average_rate else None)
    PY

本包同时保留旧 latent-only fixed-mix 的对比视频，供动作响应与画面漂移的诊断；它们不作为 RGB-consistent 视觉稳定性的证据。首帧已附在 `examples/first_frame.png`。

## 6. 动作响应旧版与 10/20 秒展示

旧 fixed-mix action adapter 另存为 `checkpoints/legacy_fixed_mix/action_adapter.pt`。必须按 [`meeting/action_vs_stability/README.md`](meeting/action_vs_stability/README.md) 的 latent-anchor / own-prefix / feedback-off 配置使用。原始 H3、旧 causal、新视觉稳定 causal 的三列视频保留完整 124 帧，包括旧版的后段漂移。

本轮长视频使用 W、seed=13、832×480、shift=2.22；243 帧=10.125 秒、481 帧=20.042 秒。稳定版配置保持不变，仅延长帧数，具体命令与每次 GPU reserve 记录在 `meeting/long_horizon/source_metrics/` 的 launch/setup JSON。

20 秒原始 H3 的 eager mask 构造曾申请一个 25.94 GiB 的临时张量并 OOM。新的可选 patch 仅编译同一 directed action mask 的构造，不改变 attention predicate 或模型：

```bash
# 从 submission 根目录执行；先按上文应用 H3 action/causal patches。
git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_long_video_mask.patch
export H3_COMPILE_BLOCK_MASK=1
python code/causal/check_long_mask.py --out outputs/long_mask_equivalence.json
```

补丁只在 `H3_COMPILE_BLOCK_MASK=1` 时启用，默认行为不变。测试逐元素比较 dense mask 和 sparse BlockMask metadata，并在同形状下替换 action assignment 后重新比较，排除 stale mask。mask 编译及不同 CPU offload 配置会影响耗时，本轮不同 run 的单次时间不能作严格吞吐排名。
