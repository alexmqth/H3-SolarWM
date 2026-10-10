# EXP-008/v2 — Worker 有限 DMD 延续训练报告

2026-10-11 HKT。状态：**新增7个cycle执行完毕，累计到cycle8；等待Judge最终能力审计及视频阶段放行**。本轮没有视频生成、额外 cycle、参数扫描或自动重跑。

## 冻结协议与恢复

按[taskbook_v2.md](taskbook_v2.md)从v1 pilot `cycle_01/`恢复student QKV、target-time、fake QKV、两个AdamW状态及CPU/三卡CUDA/训练噪声/评分噪声RNG。恢复发生在三个33B角色加载、adapter与optimizer安装完成后。teacher为冻结V3 causal模型，C1只prefill一次；student/fake每cycle以当前权重各自重建C1 KV。当前块动作从cycle2至8为**AD、AA、AD、AA、AD、AA、AD**，训练噪声来自恢复后的独立generator，不使用seed13评估噪声。每轮student保留完整8-map计算图，fake对detach端点做一次FM更新并重建KV，teacher/fake在同renoise状态评分，student进行一次DMD更新。基础权重始终冻结。

执行代码：[run_v2.py](dmd_train/run_v2.py)；[配置](dmd_train/config_v2.json)；[冻结来源清单](dmd_train/source_manifest_v2.json)，SHA `a7594031c0cae5a6e79653394b04bad8c01c88fdd07d922aa46c64c62416a0ac`。CPU预检核对29项来源、pilot配套权重/optimizer及RNG，返回`CPU_PASS_NO_GPU_AUTHORIZATION`；Judge随后签发独立[训练放行](TRAIN_RELEASE_V2.md)。实际命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-008_v3_dmd/dmd_train/run_v2.py > H3-World/outputs/EXP-008_v3_dmd_train_v2_run.log 2>&1
```

物理GPU0/2/5分别为teacher/fake/student，GPU3/4他人进程未占用。单次正常退出，无重试。原始输出 `H3-World/outputs/EXP-008_v3_dmd_train_v2/`；小证据与原stdout副本在[dmd_train/artifacts/](dmd_train/artifacts/)。

## 调用、资源与 checkpoint

| 项目 | 实际 | 获批上限 |
| --- | ---: | ---: |
| Forward | 99 = teacher C1一次 + 7×14 | 99 |
| Backward | 14 = 7 fake + 7 student | 14 |
| Optimizer update | 14 = 7 fake + 7 student | 14 |
| VAE / 视频 | 0 / 无 | 0 |
| 端到端wall | 1673.641秒（27.89分钟） | 45分钟 |
| 保守三卡GPU时间 | 1.394701 GPU小时 | 2.25 GPU小时 |
| allocated峰值 | teacher25.076、fake25.885、student28.102 GiB | 每卡44 GiB |

cycle4与cycle8各保存student QKV、student target-time、fake QKV及含两个optimizer/全部RNG的state，原件在 `H3-World/outputs/EXP-008_v3_dmd_train_v2/cycle_04/` 与 `cycle_08/`。cycle8的student optimizer step8、fake step9。配套文件SHA、metadata、127条调用事件、7轮动作顺序和每轮KV/基础权重检查的Worker CPU复核见[worker_cpu_audit.json](dmd_train/artifacts/worker_cpu_audit.json)。checkpoint不复制到Git仓库。

## 数值记录与限制

| 累计cycle | 动作 | 单轮秒 | Fake FM loss | DMD surrogate loss | Student target梯度范数 | Student QKV梯度范数 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 2 | D | 211.64 | 0.15461 | 9.395e−5 | 0.135733 | 9.37e−5 |
| 3 | A | 234.72 | 0.41085 | 3.993e−5 | 0.060488 | 2.165e−5 |
| 4 | D | 230.50 | 31.53451 | 3.39e−6 | 0.008056 | 1.55e−6 |
| 5 | A | 236.64 | 2.81548 | 4.75e−6 | 0.000443 | 约5e−8 |
| 6 | D | 243.93 | 2.53096 | 5.09e−6 | 约5e−5 | 约1e−8 |
| 7 | A | 242.49 | 2.51542 | 5.05e−6 | 约3.2e−5 | 约1e−8 |
| 8 | D | 239.52 | 2.53255 | 5.06e−6 | 约3.1e−5 | 约1e−8 |

7轮的8-map velocity梯度均有限非零，student target/QKV和fake参数每轮确实变化，缓存只读/重建、teacher复用及基础权重冻结检查均通过。但**fake loss在cycle4跃升，student参数梯度此后显著变小**；不能将数值loss趋平或训练成功当成视频改善。原因可能涉及critic追踪、DMD方向归一化或生成端点分布变化，此处不作单因素归因，也不据此追加训练。

## Worker结论

在当前三卡硬件条件下，pilot之后7轮真实33B DMD延续训练可执行且低于预算；得到配套cycle8 checkpoint。它只证明**工程训练链路**，尚未证明action、人物结构或73帧质量优于AF8/FM8。下一步仅按预定义同首窗、同噪声、同8NFE的AA/AD 73帧视频阶段验证最终cycle8；需Judge独立审核本报告及checkpoint，并签发视频marker。本轮不自动触发视频GPU。
