# AnyFlow128：同一干净历史下切换当前动作

128 clean-history A/D数值为+0.504958/−0.828582，分离度1.333539，但A存在17/34帧重置，oracle历史可能代替当前动作。用两个新增39帧/8-step诊断检验第1块的动作响应：

| teacher history/past action | held reference（已有） | switch（新增） |
|---|---|---|
| A | A→A→A | A→D→D |
| D | D→D→D | D→A→A |

四条checkpoint、initial image、基础prompt、video/audio noise、先前动作、teacher latent与anchor来源、采样配置相同。新增只改变后续action annotation；实际检查第0块5个latent逐元素不变、基础prompt及前5行action embedding不变、后续action embedding改变，其余packed layout一致。

**只对chunk1归因**：此时两分支拥有相同的已提交teacher prefix与历史动作。chunk1之后的clean commit接受改变后的action，因此chunk2的KV可能不同，不能说chunk2也只改变当前动作。RGB[17,34)仍受temporal VAE跨帧影响，首尾过渡与整片flow不能作为独立动作准确率。观察chunk1内部的运动与latent变化，不把oracle视频当free-running PASS。

保持所有训练和模型不变，GPU5/6只做两个推理。原128 generated-history D8继续，不受影响。
