# C/D 方法核查：AnyFlow composition 与 Flow Map Backward Simulation

2026-10-09 02:32补充：[旧128的512样本adaptive与输出梯度核查](../adaptive_weight_audit/README.md)完成。缩放重放正确，高noise endpoint输出梯度被明显压低；不等于参数梯度贡献、无adaptive训练对照或C效果完成。真实FM两种density的自然视频仍失败，C/D前置保持。

2026-10-09 00:57补充：[真实H3时间条件核查](../time_contract/README.md)完成148个CPU预处理用例，真实FP32时间权重、native 0–1映射、各类token时间及官方mix公式通过；另用原生Diffusers未改AST核对17个实际native时间，真实FP32权重输出逐元素相同。没有执行transformer，不能代替训练后velocity diagonal保持或C阶段完成。

2026-10-09更新：主代码已将training/validation的sigma采样与Gaussian weight参数解耦，默认精确保留旧策略；下方“当前仍绑定”的描述是核查时历史状态。[单变量真实FM分支](../fm_density_control/README.md)正在运行。D的共享角色、完整H3 FMBS及DMD loss已通过CPU图验证，见[新接通记录](../dmd_gradient/README.md)，尚未进行新33B DMD效果实验。

**这是源码核查和 CPU 数学准备，不是新的 H3 AnyFlow/Stage2 训练，不是画质验收。** 实验 B 的真实 ABot 普通 FM48 仍需完成及评估；当前没有达到“可靠 causal checkpoint”的前置条件。本目录不改变正在运行的训练架构、参数、anchor 或预算。

## 核查来源

- [AnyFlow 原论文](https://arxiv.org/abs/2605.13724)，§4.2.1、§4.2.2、Algorithm2、Eq.8、§5.4。论文网页下载到本机供核查，不随提交包复制全文。
- [NVlabs/AnyFlow](https://github.com/NVlabs/AnyFlow)，固定 commit `bf9195aa04b708a8cd9a7e742bb03265ec3a6c29`；文件 URL/hash 在 [source_files.json](source_files.json)。源码快照保留 Apache-2.0 版权头和 [LICENSE](upstream/LICENSE)。
- 本机 SolarWM commit `ce1da4e7705391eda8eeda6016c0fd3f614b975e`；具体被读取文件的内容 hash 另存，不仅凭 commit 推断未改动。

## 关键区别：SolarWM SGF 与 AnyFlow FMBS 不能混称

| 路线 | 怎样产生 student 输出 | 梯度路径与分布监督 |
|---|---|---|
| 当前 SolarWM H3 SGF | `h3_sgf_rollout` 用四个离散去噪位置，每次预测 x0 后按下一位置重新加噪；保留选定 exit 的 noisy state/x0，提交 clean raw KV | `h3_sgf_replay` 在生成的干净历史上重放 exit；bidirectional teacher/critic 对生成分布评分，DMD surrogate 更新 student |
| AnyFlow 原论文/官方 FMBS | 从选定 N-step 网格挑一个相邻区间，把完整路径换成最多三段：`1→t`、`t→r`、`r→0` | 三段非零有限映射都保留参数/状态 autograd 链，最终 x0 接 DMD；causal chunk 的历史 KV 编码在 no_grad 下进行 |
| 本项目旧 `stage2_lite_dmd.py` 主入口 | 全部 self-rollout detach 后，对生成 endpoint 重新加噪，执行单次 student velocity→x0 replay | 有独立 fake-score adapter 和 DMD surrogate，但这个 Jacobian 不是 FMBS 三段生成链；该入口 real/fake score 还沿用 chunk-cached 路径，不能因关闭新增 adapter 就称为 Original 双向 teacher |

最后一行针对当前主代码文件，不概括所有历史 outputs 中的派生脚本。旧实验里部分 paired/full-teacher 分支另有作用，须按实际调用路径判断。新的 D 阶段不得直接把旧入口改名成完整 on-policy FMBS。

源码定位：

- SolarWM `src/solarwm/backends/minimax_h3/sgf_rollout.py`：`h3_sgf_rollout`、`h3_sgf_replay`。
- SolarWM `stage2.py::score_forward` 使用 Stage0.5 bidirectional layout；`training/sgf.py::compute_sgf_kl_gradient` 与 `sgf_student_loss`。
- NVlabs [pipeline_far_wan_anyflow.py](upstream/far/pipelines/pipeline_far_wan_anyflow.py)：`inference` 将网格变为 prev/current/post 三段；`training_rollout` 自回归生成 chunk；`encode_kv_cache` 为 no_grad。
- NVlabs [trainer_far_wan_anyflow_onpolicy.py](upstream/far/trainers/trainer_far_wan_anyflow_onpolicy.py)：`generator_loss`、`discriminator_loss`、`_compute_kl_grad`；student update 之后可共同训练 forward flow-map objective。

## 已完成的 CPU 验证

[隔离 FMBS primitive](flow_map_backward_simulation.py) 接收固定条件的 noise-minus-clean velocity callback，使用 `z_r=z_t+(r-t)u(z_t,t,r)`，没有模型参数更新，也未接入 H3 trainer。

[核查脚本](verify_fmbs.py) 从固定 commit 的官方文件 AST 提取已审阅的 `inference` 和 scheduler 方法；仅去掉类型注解/装饰器以隔离外部依赖，不改方法体。用非线性小型可微 velocity 对比完整输出与参数梯度，覆盖 N=1/2/4/8/16/50、shift=1/2.22/5、首/中/末区间，共45个case。另有精确线性ODE解析解和解析梯度的独立对照。

- 45个官方对照通过；最大输出差与梯度差均约 `1.1921e-7`。
- N=1只需一个非零映射；边界区间最多两个；内部区间三个。跳过零长度段，不计不存在的网络前向。
- 负对照把第一段 detach，参数梯度差范数为 `0.282294`，验证“只保留中间 replay”不等价于完整 FMBS 梯度。
- 拒绝重复/非递减网格、缺失0端点和越界 interval 等协议错误。
- 初次 CPU harness 缺少 stdlib `copy` namespace 的失败保留；修正测试环境后通过，没有发生 H3/GPU失败。

收据：[fmbs_cpu_verification.json](fmbs_cpu_verification.json)。**这不证明真实33B上的峰值显存、训练稳定性、action preservation 或视频质量。** 后续真实实现至少要保留三段可微状态链，可通过 checkpoint/offload 管理激活；历史KV可detach，但不能把当前chunk的起始/结束shortcut随意detach后仍声称与官方相同。

## C 阶段的五项实际检查

| 要求 | 必须保持的对照 | 应记录的证据 | 当前边界 |
|---|---|---|---|
| r=t 保留瞬时 velocity | 同一个通过局部验收的 causal FM checkpoint，在安装 AnyFlow 前/后；训练后的 diagonal 再与该FM参考比较 | whole velocity 和 A/D delta、重复误差、时间MLP/LoRA参数策略 | 旧初始化已验证；不能替代未来 FM48 上的检查 |
| 时间嵌入范围正确 | H3 native time=`1-sigma`；current与target在相同native坐标，各条件行时间对不串用 | 高/中/低sigma的输入范围、embedding norm、混合后norm、每种row映射 | 代码已按官方clone+0.25 gate实现；实际新checkpoint仍待检查 |
| off-diagonal 自适应权重 | 同logical batch的FM raw loss作参考，先adaptive scale再Gaussian time weight | 每样本raw loss、scale、time weight、最终weight、时间段和样本类型；必要时记录实际梯度贡献 | 不能凭total loss下降判断困难区间学好了 |
| (t,r) 覆盖4/8步网格 | 训练采样density与loss weight分别分析；native shift网格和官方uniform网格分开 | 各实际sigma区间/endpoint样本计数，不把“低sigma少”直接判为唯一根因 | 旧128诊断已发现低噪声稀少，但增权必须受控比较 |
| flow-map composition | 同history/action/anchor及同起始z_t，比较直达F(t,q)与F(r,q)∘F(t,r)；中间z_r必须实际由前一映射生成 | endpoint差、区间归一化差；同时记录到真实latent/teacher伪标签的距离，区分内部自洽与目标正确性 | 只有可靠局部能力上才开展新的AnyFlow训练；CPU解析性质不是H3 composition证据 |

不要把 `F(z,t,t)=z` 的恒等更新当作 velocity 对角保持：步长为0时，任意错误u也会得到相同z。对角保持必须直接比较 `u(z,t,t)`。

## 时间权重的证据边界

AnyFlow原论文§5.4对比的权重中，偏高噪声的Beta(2,1.5)表现更好，uniform的test-time scaling较差；这不是“应该无条件增加低噪声权重”的证据。SolarWM H3已公开的Stage1实现则使用shift12采样和Gaussian权重。NVlabs配置/代码与SolarWM H3不是完全相同recipe，因此不能把某个仓库的默认Gaussian或shift直接说成原论文Beta实验的完全复现。

当前项目的 `training_timestep_shift` 同时进入采样变换和Gaussian权重。若后续做低噪声对照，必须明确区分改变采样density与改变loss weight；不同时改architecture、anchor或LoRA范围。

## D 阶段的正确接入条件与限制

1. B先给出可信的clean-history局部画面与当前A/D干预；C再检查有限映射。无需Stage1消除全部长时漂移，但不能跳过基本局部能力。
2. teacher必须明确为冻结Original H3 bidirectional score，critic是独立可训练fake-distribution score；两者处理同一重加噪生成样本、对应action/时间/位置条件。共享33B base可以减少参数内存，但角色切换与checkpoint反算不得误用别人的adapter。
3. student生成的当前chunk保留`1→t→r→0`的梯度链，历史chunk/CPU KV可以detach。所有shortcut内缓存保持只读，只有最终clean结果在指定位置commit。
4. 同噪声、同action/history、同N，先比较完整N步与三段shortcut产生的状态差、梯度成本与显存。近似composition不成立时，shortcut不能冒充真实N步rollout分布。
5. 官方配置示例采样N∈{2,4,8,16,50}；本项目先限定已建立的4/8步39帧协议。对于shifted网格，t/r应取实际相邻sigma，不能仍硬编码`r=t−1/N`。
6. 当前H3 callback为noise-clean，`x0=z_sigma−sigma*v`。官方DMD使用fake_x0−real_x0，经归一化构造detached surrogate；不得再次套用native H3符号翻转。critic拟合自身生成分布不能由普通teacher replay替代。

这些接入条件尚未在真实H3上完成，不将本目录的CPU结果计作D阶段成功。现阶段继续既定FM48及其匹配评测，不启动SGF/DMD训练。

主代码旧Stage2-lite入口只更正了docstring、注释与新增运行metadata，明确teacher是causal-cached原始权重、DMD使用单次endpoint replay、FMBS=false；没有改变原实验数学、没有启动该训练。真实FM48的306项冻结runtime hash保持不变。

## 21时补充：H3生成端接入

隔离primitive现已进入主代码`causal/fmbs.py`并包装真实H3 chunk_forward；[H3 CPU集成检查](../h3_fmbs_integration/README.md)在随机小模型通过六项测试，含FP32/BF16、完整三段梯度、checkpoint/offload、缓存推进后的反算、有限差分及detach负对照。仍未接入teacher/critic/DMD训练器，也未验证33B显存与视频效果；B/C前置条件有效，D未完成。
