# EXP-008 / v2 — 有限DMD延续训练与匹配视频

**当前阶段：训练已完成且Judge工程审核通过，最终cycle8视频已获独立放行；具体预算和C2崩溃停止条件见文末EVAL放行。**

2026-10-11，Judge。Pilot单cycle工程已接受，见[judge/PILOT_REVIEW.md](submission/experiments/EXP-008_v3_dmd/judge/PILOT_REVIEW.md)。本任务批准下面限定的实现与训练范围；GPU开始需CPU恢复/来源预检通过及独立`TRAIN_GPU_AUTHORIZATION.json`，视频另由Judge放行。不需要用户重复批准。

## 研究问题与基线

Research Track C，Parent为EXP-008 pilot cycle01的student/fake配套权重和优化器/RNG。检查累计8次student DMD更新后，固定8NFE下是否保留AA/AD方向和主体结构，是否存在值得继续的视觉信号。与EXP-007 AF3固定step32及EXP-006普通FM8比较。基础V3结果冻结；有限可行性与PARTIAL可接受，不能用loss宣布能力提升。

## 不变协议与恢复要求

保留Original H3 + released Action LoRA、Single I0、native timestep、own-action/action feedback、current-prefix feedback、Global、strict causal、persistent detached CPU raw KV、max_history5、12+5块、h3_fp32与现有backend。Teacher为冻结V3 causal模型；fake为last8 rank8普通FM/QKV，无target-time；student为last8 rank8 QKV + target-time gate .25。三角色独立模型/设备，完整8-map链checkpoint/offload，禁止map间detach或在反向前替换共享权重。

从`H3-World/outputs/EXP-008_v3_dmd_pilot/cycle_01/`加载student_qkv.pt、student_target.pt、fake_qkv.pt、trainer_state.pt。验证四文件SHA与元数据、student optimizer step1、fake step2；**所有模块安装和权重加载后恢复CPU/三卡CUDA RNG，以及training noise与score noise两个独立generator状态**。后续保存累计student/fake步数和cycle，不能当新初始化。

## 数据、动作与优化器

固定clean C1为EXP-006 FM8 first12、C1动作A。新增cycle2–8：偶数AD、奇数AA，包含pilot后累计A4/D4。每cycle从恢复后的独立training generator生成新C2噪声；不训练seed13评估slice。score generator继续产生fake FM和teacher/fake共享评分噪声。当前块on-policy，C1仍固定，不宣称完整多块on-policy。

Student native8step shift2.22；map为`x_r=x_t+(r-t)*v(x_t,t,r)`。score sigma .6，fake每cycle一次普通FM更新。DMD使用同detach+renoise端点的teacher/fake评分，方向`fake_x0-real_x0=sigma*(v_teacher-v_fake)`及原detach normalization。两个AdamW恢复既有state，保持lr1e-4、betas(.9,.95)、wd .01、clip1。仅pilot采用两次fake warmup；延续固定一次，不扫比例。

## 训练顺序与预算

GPU0/2/5若仍实际空闲则分别teacher/fake/student；不占用GPU3/4他人进程。

1. 启动后teacher以自身冻结权重prefill C1一次，后续复用；C1输入/动作/位置保持相同。
2. 每cycle以当前student/fake权重各prefill C1一次（2forward）。
3. Student连续8map生成当前C2，保留完整梯度（8forward）。
4. Fake消费detach端点做一次FM更新，再按新fake权重重建C1（2forward、1backward、1update）。
5. Teacher/fake对同renoise端点评分（2forward），student沿原8map反向并更新（1backward、1update）。
6. 记录梯度、参数变化与cache检查，释放当前图和更新后已失效student cache。下一cycle重建；不保存逐步大型KV。

**新增最多7cycles，到累计student cycle8；99forward=1+7×14，14backward、14optimizer update（student7、fake7）、0VAE。≤45分钟wall，保守三卡累计≤2.25GPU小时，每卡allocated≤44GiB，绝对截止09:00 HKT。** 包含加载/角色等待/保存/失败；内部checkpoint重算计时间与backward，不混成sampling NFE。使用锁与绝对alarm防止重复执行或超时。

仅保存累计cycle4和cycle8的配套student/fake权重、两个optimizer、全部RNG和source/input/parent metadata。若预算不足，允许在完整cycle边界提前收口，保存最后完整cycle，不能虚称完成8。绝不在剩余时间不足时强开一个cycle。

## 停止条件

OOM、nonfinite、base变化、角色梯度串扰、cache不一致、RNG/checkpoint混配或预算触顶时停止并提交Judge；无自动重试，不重置成本/峰值。允许普通loss波动，不能仅凭局部loss变大追加超参实验。该任务不修改time、prefix、action协议，不增加训练数据或LoRA范围。

## 视频阶段（仅准备，待Judge放行）

只选预定最后完整checkpoint，最多35forward/4VAE/0update/0.35GPU小时/44GiB。共同FM8 C1与首39 RGB，各模型自身权重建立KV；AA/AD C2同历史/动作/噪声/native8sigma，C3各自生成历史。继续使用冻结seed13初始噪声slice，不能训练噪声挑片。评估预定义顺序为prefill→AA/AD C2→AA/AD C3。复用AF3与FM8原片，不重新跑基线。

检查全部新增帧、方向与切换响应、人物结构、ghosting、chunk boundary、历史RGB不变性以及sampling/commit/decode/显存/CPU KV成本。提供DMD8与AF8/FM8并排完整73帧视频，明确共同首窗不是全程DMD首窗生成。单scene/seed不支持泛化或无限长能力。

## 交付与ROI

训练source/config/checkpoint冻结清单、实际命令、逐cycle及逐调用账本、原始stdout、配对checkpoint/RNG与Worker报告。Judge完成工程审计后再放行视频；归档至本EXP008，不能覆盖pilot或EXP007。最终无可辨联合收益就收口，不自动加cycles、换lr或扫描checkpoint。后续新问题另立任务。

# EXP-008/v2 Judge训练放行

2026-10-11，Judge。v1 pilot工程审核通过后，独立检查v2代码及CPU preflight：29项来源、配套student/fake权重、两个AdamW真实CPU恢复、RNG重放、实际DMD导入与A/D顺序通过。运行时在所有模块安装后恢复全部RNG；teacher C1复用、student/fake每cycle按自身权重重建、fake更新后再次重建，完整8map梯度保留。记录每cycle噪声SHA、缓存断言、base冻结和各卡显存。

**现在批准新增7cycles至累计8：GPU0/2/5，99forward/14backward/14update/0VAE，≤45min wall、保守三卡≤2.25GPUh，每卡allocated≤44GiB，最晚09:00。** 显卡已实时确认空闲。仅保存cycle4/8及预算提前收口的最后完整cycle，异常停止，无自动重试、不增加cycles或配置。marker绑定最终manifest SHA。

任务书taskbook_v2.md保持冻结；视频只有CPU准备授权，须训练报告经Judge审核再放行35forward/4VAE/.35GPUh。训练成功不等于生成能力通过。

# EXP-008/v2 Judge视频放行

训练已完成并经独立审核，最终cycle8固定。Judge独立CPU preflight通过，21项来源及native8step匹配，0GPU calls。**现在批准GPU0（启动前确认空闲）按prefill→AA/AD C2→AA/AD C3评估，最多35forward/4VAE/0update/.35GPUh、allocated≤44GiB、最晚09:00。** 只此checkpoint，不扫参数或重跑基线。

共同FM8 C1/首39 RGB，DMD自己prefill KV；C2同历史/动作/噪声，C3各自历史。FM8与AF3原片复用，三列视频注明实际cycle8。训练fake loss尖峰及后续较高水平作为风险如实记录，不能由surrogate降低声明改善。

先看AA/AD C2全部新增帧；若已经持续主体/场景崩溃，则停止C3、提交Judge，将当前配置作为负结果收口，不为了凑满预算生成无价值后续。普通瞬态形变/边界/ghosting仍按PARTIAL继续原计划。OOM/nonfinite/协议错误立即停，不自动重试。评估产物归H3-World/outputs/EXP-008_v3_dmd_eval，不覆盖既有证据。

## Judge提前停止 — 最终cycle8出现持续崩溃

AA C2全部17张新增帧为彩色噪声，主体与场景不可辨。停止本配置的后续推理与训练，C3 AA/AD取消。发现时AD C2进程1695658已经启动，允许其完成原预算内当前块，之后无新GPU任务；当前marker已撤销新启动资格，原授权保留EVAL_GPU_AUTHORIZATION_INITIAL.json。按实际完成的56帧证据收口，不把未测C3记为完成，不换checkpoint挑片或加训练挽救。

## 最终执行更正

AD在指令交叉时已被SIGINT中断：完成5 sampling，第6个forward中断，没有端点/视频。AD为INTERRUPTED/NOT_EVALUATED；C3未执行，不重启。最终结论见judge/FINAL_REVIEW.md。
