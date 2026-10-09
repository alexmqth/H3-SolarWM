# 同权重AnyFlow128：对角速度条件减轻视觉退化，动作仍未达标

[Original / 正常有限步预测 / 对角速度条件，完整39帧A/D](original_finite_diagonal_AD.mp4)。3列2行，全部39帧/24fps/H264/yuv420p/faststart完整解码和第30帧标签检查通过。两条全部0–38帧及Original/finite/diagonal在12/24/30/38帧对应画面已静态复核，非实时播放。

**同一128权重和8步native积分网格，将模型条件r从下一采样点改为当前t后，画质明显改善，尤其D后段严重人像/柱子叠影消失、人物与场景轮廓更完整。** 仍有模糊/轻微透明感，人物退远、运动与Original不一致；没有达到整体目标。正常finite-map生成较强动作分离，但明显更重影。该单seed对照支持“有限步预测的使用在本次rollout中加重视觉退化”，并不证明实现数学错误，也不排除其与generated-history分布偏移的交互。

| 推理条件，同一128权重 | A | D | A−D |
|---|---:|---:|---:|
| 正常finite map，r=next sigma | +0.040414 | −0.719245 | 0.759660 |
| 对角速度条件，r=current sigma | +0.074321 | −0.587728 | 0.662048 |

对角版本的动作分离反而更低，仍不足1。它是AnyFlow训练后网络的r=t消融，**不是重新训练FM的匹配对照，也不能把其视觉改善称为AnyFlow finite-map少步成功**。未替换meeting主展示。

## 实际运行审计

[actual_conditioning_audit.json](actual_conditioning_audit.json)逐项验证：两个推理使用同checkpoint路径/权重哈希、相同conditioning/packed tensor；实际积分sigma序列逐元素一致，每个chunk仍8次更新，3块24 noisy+3 commits。wrapper记录全部27次实际model time pair：noisy forward的model target_sigma确为current sigma，积分target_sigma仍为原next sigma；commit保持0→0。只有模型的目标时间条件改变，未改原runtime文件、权重、训练、anchor或solver步数。

| action | flow | E2E s | allocated MiB | CPU KV MiB | gray MAD | boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.074321 | 246.85 | 38984.16 | 6484.13 | 2.8534 | 2.9914 |
| D | -0.587728 | 247.89 | 38984.16 | 6484.13 | 2.8958 | 3.3170 |

正常finite的A/D boundary为4.1016/4.3577，对角为2.9914/3.3170。MAD仍是活动量，不能单独作画质。GPU0/3启动前都3MiB，reserve6；共享主机单次时长，不据此声称speedup。两控制器正常完成，当前无本轮GPU任务继续运行。

## 下一步

不自动增加AnyFlow训练次数，不重启已完成队列。优先在同一干净teacher history、同一noisy state上比较有限步预测与冻结同权重r=t细分轨迹对应的平均速度/端点，定位哪个区间误差大；这比继续换anchor或只看total loss更直接。若发现系统性区间失配，再以有限区间一致性辅助监督作隔离的小预算Stage1实验，并明确它与官方原loss的差别；必须回到正常finite-map 4/8步完整视频检验，不能用对角结果替代验收。尚未开始Stage2，Stage1目标保持未完成。
