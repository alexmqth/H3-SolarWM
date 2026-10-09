# 最新：64次全部视频已完成并失败

完整结论和视频见[FINAL_RESULTS.md](FINAL_RESULTS.md)。以下保留阶段记录，其中等待评测属于历史状态。

# full-history / train shift2.22：总64次训练完成，视频评测中

记录时间：2026-10-08T12:36:39.612826+08:00。第33–64次更新及前后验证共5241.37秒；GPU allocated peak=40668.18MiB。固定visual/action/time；43,237,376参数的全覆盖rank8 bank四组均更新。旧32次记录逐项保留，真实32→64回载证明仍见gpu_resume32_audit.json。

| 更新数 | 动作 | diffusion1 raw | diffusion2 raw | endpoint raw | general-map raw | weighted total |
|---:|---|---:|---:|---:|---:|---:|
| 0 | A | 0.064436 | 0.140979 | 59.624653 | 0.168970 | 0.118052 |
| 0 | D | 0.096876 | 0.198909 | 22.804770 | 0.228104 | 0.170266 |
| 16 | A | 0.064375 | 0.140852 | 61.344250 | 0.172369 | 0.117943 |
| 16 | D | 0.096801 | 0.198840 | 22.865307 | 0.225356 | 0.170182 |
| 32 | A | 0.064088 | 0.140485 | 49.927769 | 0.170048 | 0.117561 |
| 32 | D | 0.096504 | 0.198329 | 24.233921 | 0.225590 | 0.169715 |
| 64 | A | 0.061572 | 0.136058 | 47.882847 | 0.159101 | 0.113549 |
| 64 | D | 0.093433 | 0.193318 | 18.440735 | 0.213263 | 0.165033 |

验证使用相同generator seed1001，各点实际sigma/r/type/weight完全一致。没有记录/比较实际GPU noise hash；不能把预先重放noise的hash当实际采样证据。endpoint从32到64：A49.928→47.883，D24.234→18.441；公共diffusion residual也下降。raw residual改善不等于生成视频动作/画质改善，weighted total受adaptive scale影响。

64次的A/D4/8 steps per chunk按原控制器顺序生成，当前A4/D4已完成且失败，A8/D8尚未齐全；不宣称Stage1通过，不扩124f，不开始Stage2。没有更改模型、anchor、data、LR或训练/推理shift。

## 首条A4完整帧复核

A flow=-1.000683，A仍方向错误。全0–38帧contact sheet及Original/16/64的12/24/30/38帧已检查：约22帧起明显重影、后段人物与车库雾化。虽然flow较旧16次−1.248505更接近零，但画面仍失败，不能将它解释为action fidelity改善。这是静态完整帧复核，不是真人实时播放。

本条E2E=213.16s，GPU allocated peak=38984.16MiB，CPU KV=6484.13MiB，12 noisy+3 commits。conditioning与原始H3逐张量相同。单次时间不构成speedup。

[完整原视频](AnyFlow64_A_4step_raw.mp4) · [全部39帧](full_history_64_A_4step_all39.jpg) · [同帧对照](full_history_64_A_4step_comparison.jpg)。A8/D8继续，结论不提前扩展到未完成的视频。

## 4-step A/D已齐全，仍失败

D4 flow=−1.153995；与A4的分离度为0.153312，A仍错误。D全39帧与Original/16/64同帧复核也显示约22帧后明显重影、严重雾化，相比16次没有恢复可接受画面。两条视频均完整保留失败后段。

[Original / AnyFlow16 / AnyFlow64 的4步A/D完整诊断](original_anyflow16_anyflow64_4step_AD.mp4)，39f/24fps/H264/yuv420p，已完整解码。内部loss下降没有让当前4步rollout通过动作+视觉gate。
