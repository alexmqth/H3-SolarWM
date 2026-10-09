# Full-history AnyFlow step04：完整A/D8评测未通过

快照：2026-10-08T09:41:52.346187+08:00。这是4次optimizer更新的早期检查；同容量detached16参照已训练16次，不能当作等更新次数的历史梯度因果消融。GPU0的full-history16仍在训练。

[完整39帧可播放诊断：Original / detached16 / full-history4](full_history_step04_AD_screening.mp4)。视频标注更新次数不同和NOT PASSED；未替换meeting展示。

| Action | Flow | Gray MAD | Boundary RGB MAD | E2E s | GPU allocated MiB | CPU KV MiB | noisy + commit |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.308945 | 3.8492 | 5.2726 | 230.21 | 38984.16 | 6484.13 | 24+3 |
| D | -1.494894 | 3.7137 | 4.8832 | 250.18 | 38984.16 | 6484.13 | 24+3 |

A-D=0.185948，A为负、分离度低于1，gate失败。两条都是完整39f/24fps/H264，所有conditioning与对应teacher逐张量相同，真实step04 bank/time/action/full-history元数据检查通过。

A/D两条0–38帧已通过完整contact sheet查看，12/24/30/38与Original/FM16/detached16同帧比较。人物和车库大体完整但模糊，A/D均呈现近似相同的运动趋势，与detached16的8-step无明显视觉差别；仍不保真。静态全帧检查未见瞬时换场，自然运动仍需实时播放判断。

CPU KV为推理cache；不能与训练同时持有两套history的10.55GiB混为一谈。耗时为GPU1与GPU0训练并行时的单次观测，不提供speedup结论；Gray MAD不是画质分数。

同帧图：[A](full04_A_8step_comparison.jpg)、[D](full04_D_8step_comparison.jpg)。全39帧：[A](full04_A_8step_all39.jpg)、[D](full04_D_8step_all39.jpg)。
