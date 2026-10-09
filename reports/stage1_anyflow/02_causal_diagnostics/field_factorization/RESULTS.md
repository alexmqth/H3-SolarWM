# 实验A：因果化前后的整体速度与动作差分

2026-10-09 归因补充：本页 Original/causal 分支都使用 `fixed_prefix_timesteps=True`，即 text/action clean time=1。原生 H3 默认是 video time=`1−sigma`。因此下文“仅换路由”是在共同的原型时间条件下成立，不等于完整原生 H3 函数的因果化损失；原始数字和相对对照保留。E1 已将时间条件单列校准，见[机制归因限定](../../07_protocols/overviews/ACTION_MECHANISM_SUMMARY.md)。

**12个固定状态的完整对照已完成：未进行AnyFlow更新、未加载我们新增适配的Original H3，仅换causal路由，整体velocity平均余弦仍为0.9963，但动作差分平均余弦降到0.0602。动作几何的明显失配在AnyFlow训练之前就已出现。**

这支持先做真实视频causal FM桥接；不支持继续用增加AnyFlow updates作为首要修复。当前仍没有通过视频验收。

| 角色 | 整体velocity平均cos | A/D差分平均cos | 差分cos范围 |
|---|---:|---:|---:|
| Original权重＋双向 | 1.000000 | 1.000000 | 1.000000–1.000000 |
| 同一Original权重＋causal（无我们新增适配） | 0.996253 | 0.060153 | -0.087875–0.295877 |
| 此前训练的共同初始化（已有visual/action适配，bank零输出） | 0.992354 | 0.052400 | -0.032495–0.141573 |
| 共同初始化＋未训练AnyFlow r=t | 0.992354 | 0.052400 | -0.032495–0.141573 |
| 匹配的Causal FM32 | 0.992481 | 0.054469 | -0.009446–0.144178 |
| 匹配的Causal AnyFlow32 r=t | 0.992425 | 0.058434 | -0.006915–0.145463 |
| AnyFlow128 r=t（额外更新，仅补充） | 0.994578 | 0.058672 | -0.072337–0.210666 |

## 控制了哪些变量

- 固定AnyFlow128保存的A/D generated endpoints及其历史，显式加噪；后两个chunk×三个sigma×两条history。不是实际solver中间状态。
- 所有角色输入相同长度的完整prefix＋clean history＋当前noisy chunk；global positions、dual RGB anchors、prefix时间和当前A/D spans一致。只更改当前动作，过去/未来动作不变。
- 原始与causal均使用同一个SDPA后端；CPU逐元素验证Original predicate等于原生directed mask。causal保持chunk内双向、跨chunk只向过去、past/current action visibility与own-frame feedback。
- 因果部分重算完整祖先和历史action-prefix状态。这是输入范围受控的field诊断，不冒充部署时persistent-KV等价推理。当前anchor也作用于重算的历史，因此不是完整rollout质量验收。
- 原始H3-World action LoRA始终保留。第一组causal_original关闭我们新增的visual/bank/time/residual，与Original角色具有完全相同权重。
- FM32/AnyFlow32共同配置仅objective/out_dir/resume来源不同，旧审计已匹配初始化、训练动作/chunk/sigma/噪声序列和32更新。两者训练数据仍是teacher生成片段，不能算实验B的真实视频基线；等更新数不等训练算力。
- 多加入共同初始化，区分此前visual/action适配的影响；AnyFlow128不与FM32冒充同预算对照。

## 对角条件与计算后端控制

- 12个状态、两种动作下，共同初始化的AnyFlow r=t与不启用AnyFlow的FM输出逐元素相同：**True**。这只证明本初始化对角条件保持，不代表off-diagonal或训练后场正确。
- Original SDPA与原生FlexAttention的中sigma动作差分cos分别为：0.855850, 0.843377, 0.865093, 0.867620。BF16下小幅动作差分对后端误差敏感，重复同后端输出为0误差不能当作所有精度误差都为0。
- 主表全部使用统一SDPA后端，避免把这项后端差异直接算进因果化对照。结果仍限定于本组生成状态及实现。
- r<t有限区间输出的全部数据保留在原始receipt和CSV，仅作描述，不能与瞬时teacher作同语义优劣结论。

## 接下来

按目标附件继续实验B：用真实ABot视频、有效动作标注和GT历史训练普通causal FM。当前已选择6个episode、16 train＋8 validation个39帧片段；按episode隔离，真实VAE latent正在编码。B的训练与画质/动作评测尚未完成。
本次证据说明AnyFlow不是失配开始出现的唯一环节；不证明普通FM能自动修复，更不证明Stage2已具备可靠前置checkpoint。C/D仍需依次验证。

## 执行记录

```json
{
  "A": {
    "model_forwards": 112,
    "wall_seconds": 748.8056456781924,
    "gpu_allocated_peak_MiB": 24968.55615234375
  },
  "D": {
    "model_forwards": 112,
    "wall_seconds": 1155.53192205308,
    "gpu_allocated_peak_MiB": 21917.87255859375
  }
}
```

两条history共224次诊断前向，无optimizer。共享主机上的wall time只作执行记录，不是推理加速评测。GPU5/6进程已正常退出。
