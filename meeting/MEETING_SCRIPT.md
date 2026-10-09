# 5分钟答辩：用结果说明技术判断

计时包含两段视频：开场约5秒主片，后半段约20秒失败片；不另加开场历史回顾。预先打开[主片](annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)、[失败片](long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)、[指标表](METRICS.md)。完整checkpoint对应见[DEMO_PROVENANCE](DEMO_PROVENANCE.md)。

## 0:00–0:35｜先看Original vs causal

播放124帧W并排主片。

“左边是Original H3，30次整段采样；右边是因果原型，每块8步。两边首帧、动作、prompt、seed和初始噪声一致。我证明了真实H3可以按块生成并复用历史KV，但没有证明动作完整保真、长时稳定或整体加速。这个展示使用10月6日的RGB visual checkpoint，不是最新AnyFlow或E2结果。”

## 0:35–1:35｜我改了什么，为什么难

“H3-World不只是视频模型：它把每个时间位置的动作文本通过有向attention绑定到视频。我的原型把视频分成5个latent一块，当前块读取自己和过去，不能读取未来动作或视频。每块去噪结束后，再做一次clean forward，把每层raw K/V写入CPU缓存，并淘汰滑窗外历史。”

“测试能证明cache和对应重算路径一致，不能证明它等价于Original双向模型。因果化还会改变动作经多层attention传播的方式。后来的局部窗口实验保留更多Original信息流，每个sigma重算可见状态，A/D方向改善了，但人物重影仍在。这条路线没有persistent hidden KV，不能同时套用缓存加速结论。”

## 1:35–2:15｜少步为什么不等于快

“右侧是8块乘8步，共64次局部noisy forward，再加8次clean commit，不是全片8次调用。单次历史记录中Original约442到454秒，causal约673到768秒，还要支付CPU传输和RGB anchor开销，因此当前没有端到端加速。”

“SolarWM Stage0.5做双向FM适配；Stage1用因果teacher forcing和AnyFlow学习有限区间映射；Stage2用学生自身rollout、冻结teacher与可训练fake-score做DMD。KV负责历史复用，AnyFlow负责少步能力，Stage2负责生成分布适应，这三件事不能互相替代。”

## 2:15–3:15｜主动展示失败

播放完整20秒Original vs causal失败片，停在末尾。

“这和开场右侧是同一checkpoint。到10秒明显雾化，15秒后结构难辨；124帧相对稳定不代表20秒稳定。动作上也有取舍：RGB主片的A和D水平flow都为负，A方向不对；旧版方向响应更强，却更容易画面漂移。”

“我后来实际训练过AnyFlow，内部finite/diagonal一致性改善仍不足以保证视觉和动作。旧DMD-lite虽然有fake-score和self-rollout，但teacher路径与生成梯度仍是近似；后来补的FMBS、DMD方向和角色隔离只验证了小模型工程正确性，不能叫完整Stage2成功。”

## 3:15–4:15｜最后一个受控实验如何做决定

显示[最新E2局部A失败片](../reports/stage1_anyflow/01_real_video/real_transition_windows/review_step4/parking_historyD_currentA_comparison.mp4)与[最终报告](../reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md)。三列是Original权重局部N、FM-only4、FM+action4，**不是**开场两列的模型。

“我把变量收窄到训练目标。同初始化、同数据和噪声、同四次更新，只比较普通FM与加入观察动作后果排序。收齐两份停车场history和两条真实GT-history后，动作项没有一致改善；当前A的人物重影也没有消失。验证误差改善不到0.03%，不能把loss下降当成成功。所以停在四次更新，记录负结果，没有自动加到十六次。”

## 4:15–5:00｜技术判断与交付

“失败不能全部归因于缺少Stage2，因为正确reference或GT历史下仍能看到局部结构问题。值得继续研究的是：不泄漏未来的局部动作信息流，action-dependent prefix的重算规则，以及可靠的动作后果和时序监督。”

“顺序应是可信causal30步，再AnyFlow4/8步，最后研究Stage2能否缩小generated-history差距。交付里保留可播放主片与失败片、代码、checkpoint哈希、原始测量和干净环境验收。我的结论是工程可行，但控制、画质与速度尚未同时成立；清楚区分这些验收项，比继续追加模块更重要。”

## 追问备答（不计入5分钟）

- **是不是完整Stage1？** 不是效果复现成功；AnyFlow代码和真实训练做过，局部生成联合gate未过。旧展示checkpoint本身没有AnyFlow。
- **为什么不直接Stage2？** 不要求Stage1先解决所有长时漂移，但至少应有可信的局部动作和结构；当前连reference/GT history也有问题，先隔离这个缺口更有决策价值。
- **cache replay误差0有什么用？** 排除受控路径的缓存数值/生命周期错误；不证明Original等价或语义质量。
- **flow符号正确是否证明动作保真？** 不能。它是图像运动proxy，需同时看人物结构、多窗口与真实运动后果；W/S没有被这个水平指标严格验收。
- **为什么不展示最好的那条就结束？** 好片只能说明一个条件下能运行，完整失败片界定了方法的适用范围。
- **还能声称长视频效率改善吗？** 只能说实现了增量执行与历史复用；当前主片更慢、未做独占硬件重复均值，不能声称整体加速。
