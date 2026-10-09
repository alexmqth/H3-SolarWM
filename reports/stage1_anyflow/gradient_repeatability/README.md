# GPU梯度重复性诊断

GPU1确认空闲后运行同一33B初始化、A/chunk0、同noise/times的detached两次及full-no-history一次反传，不进行optimizer更新。目的是核实初始权重相同仍出现梯度微差的现象，不对画质作推断。GPU0正式训练保持运行。
