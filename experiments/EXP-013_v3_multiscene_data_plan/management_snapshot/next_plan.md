# EXP-013/v1 — 多场景 V3/AnyFlow 数据审计与训练准备

2026-10-11，Judge。**本任务只授权CPU与本地文件读写，GPU推理/编码/训练均为0。** 用户夜间持续研究授权下的下一项准备任务，09:00 HKT前交付。不能把拟议预算当作GPU授权。

## Research Track / Parent / Research Question

Branch C / 多样化训练数据准备。Parent：冻结V3 causal/native Single I0协议、EXP-007 finite-map训练工程、EXP-011原生场景输入与FM8基准、EXP-012迁移负结果。

问题：现有ABot小样本是否能支持一次来源可追溯、episode隔离、协议一致的多场景AnyFlow训练与评估？缺失哪些原生输入/教师目标、需要多少明确计算预算？当前AF2 step32工业切换PARTIAL、村落AA结构FAIL，不能靠在同一验证集上调参继续救旧checkpoint。

Hypothesis：利用已有train episodes的固定初图、重新构建native Single I0并生成冻结V3教师目标，可建立比停车场单例更有意义的训练输入；此任务仅验证数据和设计可执行，不预设训练质量改善。

## 输入与范围

审计H3-World/data/abot_bridge下clips.jsonl、annotation_pilot.json、source_metadata、preparation/encoding manifests、24个现有clip的PNG/MP4/action.npy以及必要原始源视频/动作元数据。先从实际数据重新核对数量，不把24/4train+2validation作为不可变断言。禁止下载扩展数据集，禁止覆盖旧数据或生成结果。

冻结以下评估范围：EXP-011/012工业118eb5d8b75e1b8ac23a4e9ae77af9a9、村落dfec8ed3237860eba14d67c089ecd041两整个episode及所有相关帧，不进入未来训练。它们已被观察过，称固定回归评估，不能称全新盲测集。核对EXP-007停车场训练来源与该小样本的关系；无法证明源episode关系时明确未知，不编造独立性。

## P0 数据与动作真实审计

1. 生成逐clip清单：sample/episode标识、split、原始源路径/SHA、clip/PNG/action文件SHA、source_frame_indices、source/clip FPS、实际帧数/分辨率/PTS、prompt静态来源。完整CPU解码现有小clip，检查PNG尺寸及动作数组shape/取值/列语义。
2. 核对train/validation按episode和source video hash无交叉；同episode多clip或时间重叠不是独立样本。保留现有split，不偷偷重划验证集。
3. 审计A/D命名与真实键位组合。初步看到部分片段同时包含W+A/W+D；必须报告共按键和帧级分布，不能把文件名A直接当成纯A ground truth。核对30→24FPS的帧索引采样和动作对齐，必要时使用少量原源帧/记录做CPU核查。
4. 明确39RGB/12latent片段只覆盖C1，不能冒充已有56帧C1+C2 GT目标。说明若使用真实后续帧，必须怎样从源视频及对应动作重建；本任务不自动扩剪或VAE编码。
5. 旧encoded为Dual Anchor协议，不能给V3直接复用。通过现有编码元数据和必要的少量CPU加载核实；设计新native Single I0/full37 fixture，保留静态caption而非未来叙事。

## P1 确定性候选清单与预览

在全部train episodes中，每个episode只选**target=A且src_start最小**的一张初图作为首批teacher候选；若无A，按clip_id排序选首项并解释。预期4张，不按图像美观/模型预想结果挑选。不使用validation episode。生成候选manifest及带episode/split/原动作组合标签的CPU初图联系表；完整inventory仍保留全部clip。

未来teacher反事实动作A/D是显式指定的合成控制条件，与真实录屏W+A/W+D分开标注，不能伪称复现真实游戏轨迹。候选清单冻结后，不因生成质量换图/换seed。

## P2 可执行的后续分阶段设计（仅文档/CPU配置）

提出独立任务的执行参数、调用数量、内存/磁盘/耗时估算和停止条件：

- 原生输入编码：按候选初图数量K核算3K text encoder与K image VAE encode；无视频编码。复用EXP-011协议并绑定全部来源/配置摘要。
- 教师目标：冻结V3 FM30从每个初图新生成C1，再自己clean KV分叉AA/AD C2至56。按K计算K×(30+1+2×30)=91K denoiser forward和3Kdecode；K=4为364forward/12decode。只给拟议上限和分阶段Judge门，当前不运行。异常/结构失败保留，不能自动换图；是否可用于训练由Judge决定。
- AnyFlow student：准确描述EXP-007已有finite-map目标、source/target时间采样、r=t约束和KV重建，再给出多scene batch/episode均衡方案。优先新建从V3初始化的独立student，而非覆盖step32；是否采用新初始化必须明确，不混同训练效果与模型变更。建议固定最终更新数和单一最终checkpoint，不进行验证集选择。用实测EXP-007成本估算，不用NFE直接当速度。
- 评估：普通FM8对照与student匹配8NFE、C1/动作/noise；各模型自身KV。固定两验证episode及停车场回归，区分共享历史续写与从首窗全程student。验证结果不用于调参/换checkpoint；不足以作统计泛化结论。
- 分别提出teacher数据是否可接受、训练是否启动、何时允许DMD的新门槛。当前AF4/DMD失败配置保持归档；数据准备不自动授权恢复。

建议列出“已有可用、需要新GPU产物、不能支持的结论”三栏，明确现有原始数据、旧编码、未来教师生成目标的不同用途。不追求证明大量训练会成功。

## Resource Budget / Stop Conditions

GPU调用=0，新增训练更新=0，下载=0。CPU单进程线程≤4，预计≤30分钟，新增文件≤1GiB，磁盘仍≥60GiB。只生成索引/小型预览/脚本/报告，不复制完整原始视频、权重或KV。

出现缺失源文件/无法解释的动作列或时间错位时，记录影响范围并继续可完成的独立审计；不要伪造标签或填补来源。发现训练/验证source重复则明确阻断未来训练发布，仍交付审计。无需为微小编码像素差做无穷追查；仅解决影响样本/动作/协议真实性的问题。

## Deliverables / Acceptance

独立目录下：可复跑CPU audit脚本、完整inventory/source/选图manifest、source与split/动作时间对齐审计JSON、候选初图联系表、原生fixture/teacher任务草案配置、独立AnyFlow训练/评估预算与停止条件、Worker report。大文件用原路径+SHA引用。Judge独立抽查和接受后更新next_plan/progress/archive并发布GitHub。

完成标准是可据此决定下一GPU任务及输入，不是训练成功或生成能力提升。当前不启动任何新GPU编码、教师生成、LoRA/AnyFlow/DMD训练。
