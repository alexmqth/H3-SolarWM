# EXP-011/v1 P1 Worker 结果：原生 Single I0 输入编码

P1 第三次、经 Judge 单独放行的 GPU0 尝试于 2026-10-11 06:00 HKT 完成。两张固定 ABot validation PNG 已各自从原始图像和 `caption.scene_static` 重新生成 full37/native Single I0 fixture；没有读取旧 RGB dual-anchor `encoded/*.pt`。**本阶段只是输入编码成功，没有生成视频，也没有动作/画质结论。**

| 场景 | 新 fixture SHA-256 | 文件大小 |
| --- | --- | ---: |
| industrial | `9f404c910c31e68b319cf6029d7d59fa48229d9db14df395a2890cc32057107d` | 22,121,707 bytes |
| village | `eb7590213fb7f3971f3e6d345bce185ab8f9b6fba757f43a2123b65631478775` | 22,633,909 bytes |

原文件位于 `H3-World/outputs/EXP-011_v3_scene_transfer/fixtures/{industrial,village}.pt`，大 tensor 不复制入 Git。新 fixture 的来源 PNG/文本、旧协议隔离、代码与 runtime 哈希分别见 [source manifest](source_manifest.json) 和 [code manifest](code_manifest.json)。P1 marker SHA 为 `3a95111762e8041a89060d5cb3a8e4807464d0afe2ab05ffbba1045e22cd3bc0`，采用第三次放行的显式冻结模型根目录环境，代码 manifest 仍是 `11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976`。

实际调用数：**6 text encoder forwards、2 image VAE encodes、0 video VAE encodes、0 denoiser forward、0 VAE decode、0 backward/update**；与获批预算逐项一致。第三次尝试本身耗时 61.897 秒，GPU0 `torch.cuda.max_memory_allocated` 峰值 40.576 GiB，低于 44 GiB 限制。`budget.json` 的 P1 累计计时为 **70.565/900 GPU 秒**，其中前两次加载前失败按整条 shell 命令保守计入 8.667 秒；两次失败原始日志、结果与账本均保留在 `attempt1_archive/`、`attempt2_archive/`，见 [失败1](P1_ATTEMPT1_FAILURE.md) 与 [失败2](P1_ATTEMPT2_FAILURE.md)。第三次完整日志、实际调用和 CPU 审计见 [artifacts/P1](artifacts/P1/)。

P1 之后的 [CPU fixture 审计](artifacts/P1/P1_cpu_audit.json) 为 **PASS**：

- 两场景 PNG SHA 与冻结 source manifest 一致，初始 video/audio noise 因同 seed13 逐值相同；各场景的 Single I0 只有一个 390×96 图像 anchor。
- 同场景 A/D 的非动作 prompt、噪声、audio/camera/video packed 位置逐值相同；37 个动作 span 对齐，A/D 只替换动作文本 embeddings。
- 在 stop12 与 stop17 时，未来的 25/20 个 action rows 被物理删除；当前 video row 的 direct action mask 只允许 own action，且 A/D 位置布局相同。

测试仅能证明输入与代码协议成立。G1 四个 39 帧首窗及 G2 自身历史 AA/AD 续写仍需 Judge 各自 marker，**Worker 暂未启动**。后续应按各场景/方法完整目视全部帧，不能仅凭可解码或光流值宣称迁移成功。
