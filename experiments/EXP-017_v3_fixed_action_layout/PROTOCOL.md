# EXP-017/v1 固定动作语义坐标候选

这是独立 CPU 候选协议，不修改冻结 V3 的 runtime、attention、权重或已有视频。原 packed 序列仍为 `[text | Single I0 | audio | video]`，真实 action 句子仍独立 tokenize，物理文本行连续且可变长。区别在于语义位置不再由包含未知未来动作的总 `text_len` 决定。

| 逻辑部分 | 物理行与索引 | H3原生Global三轴语义坐标来源 |
| --- | --- | --- |
| head（静态文字和图像presentation） | 当前head实际行；本候选要求与冻结同场景head长度一致 | 冻结同场景full37模板逐行复制全部三轴；head里的vision/text tags也复制 |
| 每个 action span `A_k` | 独立句子的实际token长度，连续无padding；`action_text_rows`记录当前物理边界 | 冻结模板中第k个action span的三轴坐标，广播至其所有实际token行 |
| Single I0 / anchor | 物理行随当前已知action文本长度平移 | 冻结模板的I0空间/时间三轴逐行复制 |
| audio | 物理行随文本长度平移；完整37模板固定207 audio时间位置 | 冻结模板音频三轴逐行复制，保留原w轴 |
| video `V_k` | 物理video块仍按37×390行排列，裁剪时只留已知前缀 | 冻结模板第k帧的三轴逐行复制，不以当前或未来action总长度重算 |

模板来自EXP-014同场景Original H3 full37等长A/D fixture，**先于真实未来控制输入冻结**。真实录屏只提供17个action spans；其余20个未知未来span采用fixture中的固定A句作布局占位合同，绝不读取任何真实未来动作。`visible_inputs(stop=12/17)`仍用原实现删除未来action和video物理行，返回连续、无未来padding的可见 packed；当前可见mask的行索引依现有Single-Egress规则解释。候选没有给模型偷偷增加zero prefix，也没有改动作可读范围或attention。这里保留的是H3原生三轴位置，**没有启用额外的SolarWM camera PRoPE**。

[CPU实现](fixed_layout.py)对四场景真实tokenizer分别测得前17句长度：s0每句14，s1每句13，s2每句14，s3前13句10/后4句8；canonical A/D每句10。完整逐span映射、文件SHA、物理text长度及干预测试见[CPU结果](CPU_RESULTS.json)。在全部四个冻结canonical fixture中，候选packed **所有字段**逐值等于原件；A和D在stop12/17/37的可见packed及原prompt embedding逐值相等。四个真实script场景中，改变未来句子内容和长度后，stop12/17的可见token ID、三轴position、物理索引与其他packed字段仍逐值相同；另把未来25句全部缩成单token，stop12也逐字段不变。相同已提交history的head、action、I0、audio及video语义坐标在stop12→17扩展后不变。物理行不会重复或越界，未来行由原裁剪逻辑删除。

最初候选直接调用原builder的action-aware路径，发现它在覆盖坐标之前仍用总文本长度检查`origin >= head_end`，极短未来会被错误拒绝。修订候选只复用原builder的**物理行**构造路径，再显式填写原有action span/row字段和固定语义坐标；canonical全字段仍逐值相等。第一次CPU结果和日志以`CPU_RESULTS_attempt1.json`/`CPU_RUN_attempt1.log`保留，正式结论以修订后的`CPU_RESULTS.json`为准。

**边界：** 这里比较真实句子的token IDs，未运行真实混合句子的text encoder、Token Refiner或DiT；canonical A/D的已保存embedding只用于冻结回归。原attention代码未修改，但真实变长输入下的最终backend mask、raw KV数值等价及视频动作/画质都仍是 `NOT_TESTED`。在新协议中，物理索引与语义坐标分开；即使坐标稳定，也不能推断旧KV在所有层已经可安全复用。四场景缺少真实D后果，不存在D的GT视觉验收。

后续独立GPU任务建议分门，**本轮不执行**：先在同一冻结canonical状态/sigma检查新旧layout模型输出。若旧、新两边都重新计算，2个sigma×C1/C2×2入口共**8次DiT forward**，而非4次；只有已有完全匹配的旧输出可复用，才可以另按4次新forward列账。随后独立编码真实mixed-action的head和逐句文本，保守上限14 text encoder调用，不能复用旧I0 head embedding。通过前两门后，才在第一个固定train场景用真实C1 GT-history teacher forcing做FM30与FM8 C2局部质量对照，最多1次clean commit+30+8=39 forward/2 decode；首图通过后才考虑余3图，全四图上限156 forward/8 decode、2700 GPU秒。连同canonical回归360秒、文本编码600秒，保守总上限**3660 GPU秒**，尚未授权。[Judge下一阶段草案](judge/NEXT_STAGE_DRAFT.md)给出了分阶段门槛。这里没有free-running或D GT结论；普通FM真实转移适配与AnyFlow finite-map是后续两个不同训练任务，已归档AF4/DMD配置不恢复。
