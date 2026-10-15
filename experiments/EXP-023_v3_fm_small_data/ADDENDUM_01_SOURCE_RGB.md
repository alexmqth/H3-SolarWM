# EXP-023/v1 补充01：统一全部样本的源视频RGB编码

2026-10-15 HKT，Judge批准立即执行。本补充优先于taskbook_v1中“4条旧latent复用/30新窗口编码”的规定，其余研究问题、选样、训练和评测不变。

## 依据与修正

Worker发现EXP-016 run_vae.py读取EXP-015重编码real56.mp4；Judge核对源码确认。旧4条latent并非直接来自同一source RGB解码。Worker报告pilot中source RGB与重编码展示片有非零像素差异，精确MAD以其CPU证据为准。为避免在本轮32train中混用两个像素预处理流程，**34个所选clip全部从各自原始源视频按冻结source_indices直接解码，构建I0/full17和对应条件**。源视频本身可有压缩；这里的统一来源不意味着无损相机原始素材。

- 32train与2dev IDs、动作、源索引保持SELECTION/EXP-021清单不变。
- 原4train不再复用EXP-016 latent或旧I0 anchor；新增其4×(full17+I0)=8次VAE encode，重新构建并核对相关fixture、anchor与initial noise。纯文本embedding只有文本/编码器SHA相同且无图像耦合时可复用，位置与I0必须重建核查。
- 30条原计划编码按原步骤继续；已从正确source RGB完成的编码可复用，不重跑。两个pilot保持，prefix12比较预算不增加。
- 不覆盖旧EXP-016/019/022文件。EXP-022既有样本仅用于独立的训练wrapper回归检查，不成为本轮混用数据输入。新编码与旧编码有差异是来源修正的预期，不要求其数值一致。
- 本轮checkpoint0与128updates始终使用统一后的相同输入，来源SHA写入新manifest和checkpoint metadata；最终报告注明与旧EXP-019视频并非完全相同像素输入。

## 修订预算与直接执行授权

VAE encode总上限由62增为**70**；编码阶段GPU秒由3600增为**4200**（含pilot≤900）；整个任务GPU秒由13200增为**13800**（3.8333GPU小时）。新增600秒仅供额外4clip编码及条件准备，训练≤6000、评测≤3600不变。600text、12decode、440顶层DiT、128backward/128updates、最多2空闲卡/训练1卡、RAM/磁盘及硬停规则全部保持。

Worker收到后立即按本补充执行，更新CPU preflight/配置/合并预算与来源manifest，将原AUTHORIZATION及本补充授权共同绑定；检查通过后连续进入编码、训练和评测，无需额外marker。单一最终预算账本必须包含已消耗调用，不能重置计数。若仍有预处理来源不明，停止受影响样本并报告，不自行混用输入。
