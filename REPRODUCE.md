> **展示材料整理（2026-10-10）：** 现有运行路径、权重和adapter接口保持原样。新[report](report/README.md)可单独复制汇报，其中code是阅读快照，不是独立33B环境；[素材制作与完整性验收](archive/reorganization_20261010/README.md)只使用CPU/PyAV，不运行模型。

# 最终复现说明

本包不包含33B底座、发布的H3-World基础LoRA或原始数据。**最小验收不启动E2、AnyFlow或DMD训练**：重建环境/源码，检查KV，运行随机小H3类的训练smoke，使用真实底座与包内旧RGB adapters生成39帧并输出指标。

主片来源见[DEMO_PROVENANCE](meeting/DEMO_PROVENANCE.md)，当前质量结论见[实验报告](docs/EXPERIMENT_REPORT.md)。复现通过仅证明工程可执行，不代表动作/画质通过。

## 1. 干净Python环境

以下命令用 **bash**，从本仓库根目录执行。需要Python3.10、CUDA兼容驱动；验证机器是48GiB级L40，通过CPU offload运行33B。CPU RAM也需足够容纳底座、文本编码器及缓存，不是只需一张GPU显存。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.1
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

升级pip是必要的：本次新venv自带旧pip解析`typing_extensions`元数据失败，升级后通过。不使用`--system-site-packages`，不把原研究代码目录加入`PYTHONPATH`。最终安装版本完整记录在[依赖锁定清单](reports/final_acceptance/pip_freeze.txt)；需要固定传递依赖时，在先安装上述CUDA版torch后用该清单安装。

## 2. 显式准备外部权重与干净源码

准备合法获取的两个外部资源，不需要完整ABot数据集：

- [MiniMax/MiniMax-H3](https://huggingface.co/MiniMax/MiniMax-H3)的`FL2VA`，包含transformer、text_encoder及各自shard index，video_vae/source、audio_vae、processor及相应配置/分词文件。
- [H3-World](https://github.com/Danzer1xxxxChan/H3-World)发布的`step-10000.safetensors` action LoRA。验证版本SHA256：`ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526`。

设置你自己机器上的路径；不要照抄其他用户的目录：

```bash
export H3_BASE_DIR=/absolute/path/to/MiniMax-H3
export H3_ACTION_CKPT=/absolute/path/to/H3-World/step-10000.safetensors
python scripts/prepare_runtime.py \
  --minimax-model-dir "$H3_BASE_DIR" \
  --h3-checkpoint "$H3_ACTION_CKPT" \
  --out outputs/runtime_preflight.json
source env.sh
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

脚本只克隆Git源码，**不自动下载模型**。它检出`code/diffsynth_base_commit.txt`中的`300e3e4…`，依次应用H3 action、causal、long-video mask、AnyFlow、native-FP32五份补丁；后两者不表示本次推理启用AnyFlow/FP32。默认仍用旧checkpoint所需的`legacy`精度。

脚本验证shard index与safetensors键、发布LoRA格式、包内两份adapter及首帧的哈希，并写出依赖收据；底座不做整套文件hash，实际张量由推理加载验证。它只在新DiffSynth checkout内创建指向现有底座的只读使用路径。源码已存在但不是规定版本/补丁组合时会报错，请换一个新checkout，而不是在研究环境上叠加补丁。

本包附带`examples/first_frame.png`。benchmark现在支持显式`--h3-checkpoint`、`--first-frame`和`--scene-prompt`，不会再要求原实验目录存在。H3 action LoRA应加载104对；加载数量不完整会拒绝继续。

## 3. KV cache测试与训练smoke（CPU即可）

```bash
python -m pytest -q tests/test_h3_cached.py tests/test_local_topology.py \
  --junitxml=outputs/kv_and_causality.xml
python code/causal/train_smoke.py --device cpu --steps 8 \
  --out outputs/training_smoke.json
```

测试包含真实`MiniMaxH3DiT`小配置的多层cached/recompute比较、滑窗淘汰、非整块尾部、非法重复提交和动态anchor重放，以及局部拓扑的未来不可见检查。训练smoke是**随机初始化的小H3、固定合成latent、普通FM和LoRA**，要求loss/gradient有限、参数实际更新、固定批次loss下降；它不是33B预训练效果测试，不重启研究训练，也不覆盖完整AnyFlow/Stage2。

## 4. 真实权重最小causal inference（39帧）

先看`nvidia-smi`选择有余量的卡。`ABOT_VRAM_RESERVE_GIB`从GPU**总显存**扣除得到模型驻留预算，不是对整个进程峰值的硬限制。共享GPU时要为其他进程和本次激活保留余量。本次验收GPU2已有其他负载，使用reserve20；该设置不能拿去和旧reserve5的耗时作速度排名。

```bash
export CUDA_VISIBLE_DEVICES=2    # 按当前机器实际空闲卡修改
export ABOT_VRAM_RESERVE_GIB=20
python code/causal/benchmark.py \
  --modes cached --steps 8 --flow-shift 2.22 \
  --num-frames 39 --chunk-frames 5 --history-chunks 5 \
  --action-preset A --seed 13 --cache-device cpu \
  --history-source generated --anchor-mode dynamic_last_frame_rgb_dual \
  --h3-checkpoint "$H3_ACTION_CKPT" --first-frame examples/first_frame.png \
  --scene-prompt 'A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage.' \
  --causal-adapter checkpoints/visual_rgb_tail16/causal_adapter.pt \
  --causal-action-adapter checkpoints/visual_rgb_tail16/action_adapter.pt \
  --action-prefix-mode causal --action-feedback \
  --out-dir outputs/acceptance_A39
python scripts/verify_inference.py outputs/acceptance_A39
```

应生成：

| 文件 | 验收内容 |
|---|---|
| `cached.mp4` | 39帧、832×480、24fps、H264/yuv420p，全帧可解码 |
| `setup.json` | 实际外部权重/首帧/adapter路径与SHA256、LoRA加载数、源码位置、初始噪声hash |
| `cached.json` | status complete，3chunks×8=24noisy forwards＋3clean commits、CPU KV、GPU allocated峰值、分段耗时 |
| `acceptance.json` | 上述断言、完整解码、灰度相邻MAD、近似17/34帧boundary MAD、Farneback响应、整段/内部首chunk耗时 |

两份adapter必须来自同一`checkpoints/visual_rgb_tail16/`，哈希对照`meeting/DEMO_PROVENANCE.json`。不要混用`legacy_fixed_mix`、`action_diagnostic`或AnyFlow checkpoint。该A样本是可执行性验收，方向/质量仍可能失败；检查脚本不把它误判为质量PASS。

## 5. 124帧对比与性能表的边界

已有完整会议片和测量见[meeting/METRICS.md](meeting/METRICS.md)，不必重新训练。若以后重新测量，将causal命令改成124帧；baseline单独运行：

```bash
python code/causal/benchmark.py \
  --modes baseline --baseline-steps 30 --flow-shift 2.22 \
  --num-frames 124 --action-preset A --seed 13 \
  --h3-checkpoint "$H3_ACTION_CKPT" --first-frame examples/first_frame.png \
  --out-dir outputs/baseline_A124
```

baseline不加载额外causal/action residual，不启用dual anchor。左右保持同首帧/prompt/actions/seed/分辨率并核对video/audio noise hash。严格速度表还需相同GPU、相同offload/编译设置、无其他计算负载、一次warmup后至少3次测量，报告均值与离散程度。8steps/chunk不是8次生成全片；首内部chunk时间不是端到端首帧播放延迟。

当前收尾只做最小推理与测试；共享硬件条件下不追加新的124帧性能排名，不编造未测过的显存组件分解、LPIPS/FVD或warmup均值。

## 6. 高级研究分支只作档案

[AnyFlow文档](docs/stage1/STAGE1_ANYFLOW.md)及[完整进度](docs/archive/PROJECT_PROGRESS.md)保留历史命令与结果，历史“运行中”段落是时间记录，以当前首页与最终报告为准。E2已冻结在4更新，不运行`train_stage1_anyflow.py`、在线replay、`stage2_lite_dmd.py`或E2队列来完成本次验收。E2的局部N没有persistent hidden KV，不能拿本节旧KV benchmark替代E2协议。

最终验收结果与环境收据见[reports/final_acceptance/README.md](reports/final_acceptance/README.md)。
