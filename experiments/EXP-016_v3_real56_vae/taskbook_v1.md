# EXP-016/v1 — 真实56帧VAE编码与前缀因果性

2026-10-11 08:40 HKT，Judge。**P0仅CPU实现立即授权；P1 GPU待Judge绑定源码/配置/源清单的marker。** EXP-015已验收数据，有完整224帧/动作/来源证据。用户夜间研究授权继续有效，不需要重新请求用户许可。此任务不启动DiT或训练。

## Research Track / Parent / Question / Hypothesis

Branch B/C训练数据工程；Parent EXP-015新真实56帧及冻结V3 source video VAE。问题：相同实际新视频的56帧编码前12latent是否与其前39帧单独编码一致，并能形成有来源的SingleI0、C1/C2 GT latent和可辨的VAE重建？假设原生causal_encoder/temporal-isolated norm成立，普通VAE有损重建可接受。

Baseline是同一新56RGB的前39，严禁用旧39clip比较。新变量仅输入长度39/56；模型/精度/预处理/空间tile固定。

## Inputs / Controlled Variables / Out of Scope

EXP-015 OUTPUT_MANIFEST.json SHA adbd82f57f82a17cd0b830192d809b5e524922a91e2ffbca508ac2173775f051的四图/视频，固定顺序43866101、7199292c、9dc2e588、b784d995；实际原始文件SHA逐项核对。视频832×480/24fps，取真实decode RGB，输入float32[0,1]遵循native preprocess_video(min_value=0)；I0采用该视频新PNG，native preprocess_image/min_value=0/process_image=True。VAE computation bfloat16、tile256/overlap64、source权重与EXP-014同路径/摘要；不修改VAE或用旧Dual Anchor。

不加载text encoder/DiT/audio VAE，不构造prompt/packed/noise，不改action/prefix/time/Global RoPE，不训练或生成新反事实。EXP-015变长action-layout风险单独保留待解，不用padding/重定位暗中修复。

## P0 — CPU入口准备（立即执行）

复用已验证model加载环境，显式DIFFSYNTH_MODEL_BASE_PATH；仅加载video VAE，预检确保不调用GPU。建立源码/配置/source manifest、no-marker拒绝入口、输出不覆盖、逐调用write-ahead计账、截止时间/磁盘/峰值门。图像预处理必须与native同一取值范围和resize规则；video56与39输入的首39逐值同源。

只检查新入口实际风险：4source身份/长度/帧率、0..1输入、39/56分块数量、source VAE causal flag和time-isolated norm的CPU源码依据、预算。CPU通过后Judge发P1 marker；不得自主改模型/重试GPU。

## P1 — 单卡顺序有限编码（待marker）

每图依次：
1. 新I0 process_image=True encode一次，保存[1,24,1,30,52]。
2. 完整56RGB process_image=False encode一次，保存[1,24,17,30,52]。
3. 同一RGB数组前39 encode一次，保存[1,24,12,30,52]。
4. 56编码前12与39编码比较exact_equal、max_abs/mean_abs、relative_RMS、dtype、runtime VAE flags/normalizer类型。先报告，不为差异自动重算或切fp32。非zero若超过max_abs0.02或relative_RMS1e-3暂停后续scene，由Judge判断数值误差还是协议问题；小数值差异不自动宣称严格等价。
5. 只decode一次完整17latent并保存原生56RGB重建、source/reconstruction并排视频、每帧MAD/PSNR。保留原生decode过程和实际输出帧数，不重复图像/39decode。检查finite和重建主体/场景，普通纹理/柔化为PARTIAL；持续结构瓦解/全彩噪停止后续。

每图1image encode+2video encode+1decode；四图共**4image encode、8video encode、4decode、0text、0denoiser、0backward/0update**。视频GT表示真实观察；SingleI0图像latent不是把视频latent首token改名。

## Resource Budget / Deadline / Stop

- 仅启动时真正空闲的一张GPU0（如不空闲，报告Judge重定卡），不动GPU3/4。整个P1上限600GPU秒，含加载/保存/失败成本；allocated峰≤44GiB。
- P1必须08:50前启动；每scene启动前检查剩余总预算与08:57截止，08:57停止新增模型调用并交付已有结果。09:00项目最多3卡，本任务至多1卡且结束即释放。
- CPU runnable限制在4核affinity，线程参数≤4；新增文件≤1GiB、剩余磁盘≥60GiB。大型latent.pt只放新raw输出目录及SHA，不复制权重。
- 错来源/非有限/OOM/预算超限/帧数不符/输入前缀不同/显著前缀差异即停，不重试、不加scene或额外tail扰动编码。若P0来不及08:50，保留可审阅代码与未执行状态，不赶任务开跑。

## Deliverables / Acceptance

独立EXP-016代码/配置/来源与runtime清单、CPU检查、marker、真实命令与账本、逐scene端点/重建视频、输入range和前39同源证据、实际flag/norm、前12数值比较、forward/encode/decode计数、GPU wall/peak/存储、重建指标和边界图。Judge核验真实tensor与全部重建视频；只验收编码/重建可行性，不是FM/AnyFlow生成PASS，也不解决完整action packed协议。

任务完成后等待Judge最终收口；完整训练、AnyFlow或DMD不在授权内。下一步优先提出可复现的action-content-independent layout候选及冻结V3回归门，再讨论真实数据适配训练。
