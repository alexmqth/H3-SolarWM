# EXP-015/v1 — 四训练episode真实56帧与native动作准备

2026-10-11 08:27 HKT，Judge。**仅CPU执行立即授权；GPU编码、推理、训练均0。** 前任务EXP-014目标生成已验收、四场景动作正确监督未成立，最终证据已归档。用户已授权Judge夜间持续推进；本任务无需重复请求用户确认。

## Research Track / Parent / Research Question / Hypothesis

Branch B/C数据准备；Parent EXP-013固定train候选、EXP-014教师质量结论及原ABot action转换代码。核心问题：现有四train episode能否直接提供有来源、时间对齐明确的真实C1+C2共56帧数据，作为后续native V3训练候选？假设现有原始录像足够且native action映射可追溯；数据准备通过不代表模型训练/生成通过。

## Inputs / Baseline / Controlled Variables

固定EXP-013 candidate_manifest.json SHA b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6的四train episode及src_start：43866101158a538d22c52ac5b2fa606b/1245、7199292ca430b478bfd0d83cf2e519d8/1670、9dc2e5882a294b8d68d6b2b31ff3ef65/1410、b784d995827f53977a8067e5b9dc4b5e/1635。不得换片/起点/seed、扩scene或下载新数据。排除全部validation episode；保留已观察validation非盲测限制。

Baseline为原39帧真实clip，唯一数据范围变化是延长为56帧。source30fps→24fps选择固定为src_start+5*(j//4)+j%4，j=0..55；原scale/crop832×480、FFmpeg select/setpts N/24、x264CRF14/veryfast/yuv420p复用。源视频与annotations SHA、实际命令/版本、静态caption来源必须记录。

## Execution Plan / Changed Variables

1. CPU预检四个源/annotations及train归属、所有frame index在范围内；建立新独立输出目录和manifest，不覆盖原39clip/编码/EXP-014证据。
2. 从完整source抽取四段56RGB，24fps；完整解码224帧、PTS递增和源时间对齐检查。保存新I0 PNG。重编码可能改变前39帧像素；明确记录新I0和EXP-013旧PNG的SHA/MAD，不拼接旧39帧强制相等，不宣称fixture字节相同。
3. 由原annotations和COLMAP按原归一化生成四份56×17 raw action，所有11原始key逐行对源；保留连续camera/motion列、episode尺度与caption，不把真实A+S/J/L改成纯A或D。
4. 审核native非均匀pooling：FRAME_PER_TOKEN=(1,4,4,4,4)，56RGB→17latent。先完整56→17再按12+5切C1/C2；首12span恰到39，第13开始39，不可独立C2重置相位。核对key max、连续sum/scale/clip及与原39数据首12的关系。
5. 单独记录action_script映射：KEYS9的F是从J/L与COLMAP yaw派生的fast-pan，不等于Space；Q/E/Space如非零必须披露。列出原始混合key、相反key被purify的span、camera_rate短span平滑的可见范围。验证C1最终conditioning不读取C2数据；若转换不可无损表达，保留原raw标签并标PARTIAL，不改协议掩盖问题。
6. 提交后续native Single I0和video VAE编码设计（本轮不运行）：真实C1 teacher forcing与合成teacher C1/student自生成历史分别定义，student参数更新后重建自身KV；普通FM训练与AnyFlow target-time finite-map目标区分。不得复用旧Dual Anchor编码。明确训练candidate只包含真实记录动作，没有反事实D GT；不能声称这些纯A为主数据已解决D覆盖。

## Resource Budget / Stop Conditions

- CPU同时最多4线程；CUDA_VISIBLE_DEVICES=''；0 GPU调用/0模型编码/0forward/0backward/0训练。
- 新增数据≤1GiB，剩余磁盘≥60GiB；执行墙钟预算20分钟，08:50前停止新处理并交付实际结果，09:00监督截止。
- 来源/分割/时间不一致、frame越界、输出覆盖风险、预算超限即停止并报告；不扩下载/scene/做模型测试。不以普通像素编码差异反复重建，使用有来源的明确容差和原索引证据。

## Evaluation / Deliverables

独立EXP-015目录：实现代码/冻结配置、来源与输出manifest、实际命令和CPU运行日志、四56帧视频/I0/raw action路径与摘要、全部224帧解码/PTS/源索引核对、11keys逐行验证、17latent spans/keys9/action script与信息损失报告、C1未来隔离检查、资源实际值、后续编码/训练设计。Git可放小视频/图像/JSON/代码，numpy大tensor留原位置+SHA。

Judge验收数据可追溯性/对齐/实际可表达动作及限制，不把准备完成当训练成功；ROI尺度接受有解释的编码MAD和真实遮挡。提交report.md后等待Judge收口。不得自行启动GPU、AnyFlow、DMD、seed搜索或teacher补救。
