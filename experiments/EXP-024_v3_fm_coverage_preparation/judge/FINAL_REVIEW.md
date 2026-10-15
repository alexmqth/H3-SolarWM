# EXP-024/v2 最终审核

2026-10-15，Judge：**accepted，训练工程可执行性/资源profile通过，视频能力未测。**

独立核对源码/授权/输入SHA、250条route trace顺序、C1/C2各50层有限非零梯度、冻结参数无梯度、3forward/2backward/2updates完成和exit0。两次optimizer均在路由与梯度门控后执行，C2完整CPU K/V/RoPE摘要检查通过。76.366993 GPU进程秒，allocated峰24.163439GiB、reserved峰27.507813GiB、采样点RSS最大55.736687GiB；6.332GiB CPU历史KV。见[独立复核](FINAL_AUDIT.json)。进程记录中的EXP-024/v1是沿用CPU runner标签，GPU实际依据v2任务书/授权；保留原始日志，不修改历史标签。

只有一条样本、固定sigma=.5、C1/C2各一次更新，不能用两个loss或梯度范数推断学习收益。full50的profile峰值与旧tail8不同样本/步骤状态，不能宣称扩大参数反而降低显存。

下一项EXP-025/v1：全层rank8、C1/C2均覆盖的32样本×4遍（256updates）匹配前后评测。选择四遍是保持每clip每block曝光可解释，避免再次只跑2/64updates就评价路线；仍为有硬预算的小数据拟合，不叫充分训练。全层容量与块覆盖同时变化，是联合基线而非单因素消融。
