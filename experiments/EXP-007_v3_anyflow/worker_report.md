# EXP-007/v2 Worker 报告 — AF1 单次真实模型更新

`task_id=EXP-007` · `plan_version=2` · `worker_status=complete_pending_judge` · `capability_claim=engineering_warmup_only`。任务依据 [冻结任务书](taskbook_v2.md) 与 [一次修复重跑授权](AF1_RETRY_AUTHORIZATION.md)。原始结果、逐调用账本和 stdout 的小型副本见 [artifacts/attempt1](artifacts/attempt1) 与 [artifacts/attempt2](artifacts/attempt2)；完整输出和两组 checkpoint 保留在 `H3-World/outputs/EXP-007_v3_anyflow_af1_attempt2/`，不进入 Git。

## 协议和实际命令

从 Original H3 + released action LoRA 起步，使用 V3 strict chunk-causal/current-prefix/Single I0/native time/Global RoPE、首 12 后 5 latent。训练端点是冻结 V3 的 30-step generated C1 `first12_A.pt` 和 EXP-002 AA C2 `chunk_12_17.pt`；不是 GT 或独立测试集。新 student 在最后 8 层 QKV 安装 rank8 零初始化增量，target-time MLP 从实际加载 H3 time MLP 克隆，gate 0.25；base 与 released LoRA 冻结。C1 clean history KV 在安装后由 student 自己重新建立。

CPU 命令：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-007_v3_anyflow/run_af1.py --preflight`，source/hash/shape 检查 PASS；AF0 tiny-H3 原预检也再次 PASS。GPU 命令：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-007_v3_anyflow/run_af1.py --gpu 1 > H3-World/outputs/EXP-007_af1_attempt2.log 2>&1`。使用 GPU1，source/config/runner SHA 已冻结于 [attempt2 manifest](source_manifest_v2.json)。CPU generator seed 170007；AdamW lr1e-4、betas(0.9,0.95)、weight decay0.01、grad clip1；物理 batch1、logical batch4。

Attempt1 在 `sha(__file__)` 类型错误处停止，尚未加载 33B 或执行 forward：0 forward / 0 backward / 0 update，3.538686514 GPU 秒。[事故记录](AF1_ATTEMPT1_FAILURE.md) 保留失败原因。Judge 批准一次实现修复重跑；attempt2 使用独立目录，保留原上限并从 0.75 GPUh 中扣除首次耗时。

## Attempt2 实际证据

| 项目 | 结果 |
|---|---:|
| 对角初始化，相同 C2 noisy state max abs / relative RMS | 0 / 0，`allclose(1e-5)` PASS |
| 非对角 target-time 输出 | finite；相对对角 RMS 0.13984 |
| 训练参数 | 17,555,328（末8层 QKV + target MLP） |
| 逻辑 batch | 2 diagonal/FM + 1 endpoint + 1 general map |
| 真实调用 | 22 forward、4 backward、1 optimizer update、0 VAE |
| 梯度 norm | target 0.143265；QKV 0.0000820；均有限非零 |
| 参数变化 | target 与 QKV 均变化；冻结 base 版本未变 |
| 更新后历史 KV | 重新 clean commit，KV 相对 step0 变化；旧 cache 只读且 detached |
| checkpoint | step_00 与 step_01 target/QKV/optimizer/RNG 全部保存；CPU 重载和嵌入 SHA 核对通过 |
| 峰值 allocated / reserved | 26.631 / 27.102 GiB |
| GPU 时间 | attempt2 183.043 秒；含 attempt1 总 186.582 秒 = 0.05183 GPUh |

四个样本的 `(sigma,r,raw_loss)` 分别为 `(0.770971,0.770971,0.129367)`、`(0.062257,0.062257,0.504157)`、`(0.863774,0,9.922815)`、`(0.151869,0.063276,0.410067)`。endpoint raw loss 较大，但按冻结的 adaptive scaling 后 weighted loss 为 0.218893；四项完整权重和数值在 `result.json`。训练中每样本 3 detached + 1 gradient forward，4 次 backward 后仅一次 update。更新后重建 C1 KV 的第 22 次 forward 也计入账本。

## 判断边界

本轮验证 **V3 协议上的新 target-time student 可以初始化、反传、完成一次 finite-map 更新并保存可重载 checkpoint**。它没有做 4/8-step 视频推理、独立噪声/动作评估或画质比较；一个更新不能证明 AnyFlow 少步质量改善，也不能作为 Stage1 成功验收。后续训练及评测只能按 Judge 新任务书执行；不能自动从本轮扩到 32 updates 或 DMD。
