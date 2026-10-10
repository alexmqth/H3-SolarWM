# EXP-011 P1 retry

## P1首发失败与一次修复重跑授权（05:56 HKT）

首发在模型加载前因sha字符串路径类型失败，0模型调用；保守计费4.386365982秒，原始日志/账本/首发源码与marker已归档并核对哈希。修复统一Path(path)，初始化纳入异常处理，独立test_p0通过。Judge批准P1 attempt2一次重跑，仍6text/2image encode、900秒累计预算、0去噪/0decode；失败成本不重置。代码manifest现为11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976，无协议变动。G1/G2仍未授权。
