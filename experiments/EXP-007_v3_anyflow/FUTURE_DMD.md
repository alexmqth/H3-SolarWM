# V3后续DMD：实现核查与下一阶段条件

2026-10-11，Judge预审。当前是设计准备，GPU执行需后续任务编号/预算。用户已授权夜间持续研究，Judge会根据V3-AF实际结果发布任务。

## 复用与拒绝混用

`H3-World/code/causal/dmd.py`的noise-minus-clean公式可复用：`z_sigma=(1-sigma)*x0+sigma*noise`，`x0_hat=z_sigma-sigma*v`，因此DMD方向为`fake_x0-real_x0=sigma*(v_real-v_fake)`。teacher/fake必须在同一个detach后重新加噪的student endpoint、相同sigma/条件上评分。

旧`stage2_lite_dmd.py`采用旧chunk_forward/anchor协议，并用重加噪endpoint的单次x0 replay作为可微输出。它没有对真实多步student rollout求导，不能直接称为V3下的on-policy多步DMD，也不能作为本轮已经完成的证据。

## 建议最小独立任务

1. 从通过有限视频检查的V3-AF checkpoint开始。明确冻结V3作为teacher；teacher采用同V3 causal条件和独立teacher KV，不冒称原生全双向H3/SolarWM teacher。
2. 新建独立fake-score模型/adapter与optimizer，在当前student实际生成端点的分布上做普通FM拟合。fake不复用generator target-time输出冒充score；teacher、fake、student各自按其权重构建同clean-history输入下的KV。
3. 先少量fake warmup，再有限generator/fake交替更新。generator loss回传到**真实有限步生成链**（首轮可只训练第二块、历史detach），不能只对另一个endpoint replay求导并称完整rollout梯度。
4. teacher/fake score、重新加噪样本和normalizer均detach；仅student实际endpoint保持计算图。不得通过切断每个map之间的图来节省显存却保留“全链”表述。
5. teacher/fake使用普通velocity，target-time student使用显式(sigma,r)有限映射；三个角色的时间符号、native prefix、own-action/current-prefix及Global位置分别核查。
6. CPU先确认方向符号、角色梯度隔离和真实多步图；再一个真实GPU cycle测显存/耗时；根据成本设有限总cycles和匹配8NFE视频预算。不能因为有空闲卡就无限增加critic更新或无效cycle。

## 验收尺度

工程通过要求真实student samples、独立fake更新、正确DMD方向、generator有效梯度、角色/历史来源可追溯。能力判断仍看匹配动作/噪声/历史条件的视频、人物结构、边界和成本；loss下降或direction非零只支持机制运行。少量循环没有可辨收益可停止并归档，不把训练步数当成功。

本阶段若使用冻结C1 clean history，只能称当前块on-policy训练；未来完整多块student历史分布与真实跨块梯度需要另计预算。不宣称完整SolarWM Stage2复现。
