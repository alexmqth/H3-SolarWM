# EXP-008/v1 DMD pilot Judge审核

2026-10-11 03:45 HKT。**接受一个真实33B DMD cycle的工程可行性。** 原进程849858已退出，外部result为complete_pending_judge、budget正常关闭；尚无DMD视频或质量收益结论。

## 实际计算与独立证据

GPU0/2/5分别teacher/fake/student。17显式forward、3backward、3update、0VAE。完整占用254.898624秒，保守三卡成本0.21241552GPU小时；运行内部result的246.33秒不含全部入口成本，正式成本采用budget总墙钟。Allocated峰值teacher25.07004GiB、fake25.87917GiB、student27.96513GiB，低于44GiB限制。

Judge运行[audit_pilot.py](audit_pilot.py)，[独立结果](pilot_audit.json)全部通过：23项来源未改、严格角色/call顺序、native8map时间序列、相同sigma .6评分、全部8个map梯度有限非零、两次fake更新与KV重建、student target/QKV均变化、保存权重确实不同于AF2父权重、配套四文件SHA与元数据、student optimizer step1/fake step2及CPU/三卡/训练/评分RNG。Base冻结、score梯度隔离和历史KV不变性由已审核实际runner断言佐证；没有独立重跑GPU。

8map velocity梯度范数约4.98e-6至2.57e-5，DMD direction RMS0.017188。两次fake FM loss约0.11061、0.10823。该数值仅证明本次计算链有效，不能代表fake-score充分拟合、teacher更优或视频提升。

## 范围与下一步

Teacher为冻结V3 causal模型，fake为独立普通FM/QKV，student从AF2 step32继承target/QKV。使用固定FM8 clean C1与独立AA C2 training noise；不是完整多块self-history on-policy或SolarWM复现。Student的完整8map链保留梯度，未detach map间连接。

实测成本允许一次有限延续：拟新增7cycles到累计8，然后用固定final checkpoint做与AF3/FM8匹配的视频。固定配置，不扫学习率、critic比例或checkpoint；若没有可辨联合收益则收口。新任务书和独立预算须在运行前冻结，pilot本身不自动授权后续训练。
