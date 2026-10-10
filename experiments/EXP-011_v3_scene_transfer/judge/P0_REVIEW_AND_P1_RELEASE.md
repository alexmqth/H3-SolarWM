# EXP-011 P0审核与P1输入编码放行

2026-10-11 05:53 HKT。Judge独立CPU入口预检与test_p0均PASS，最终代码清单26项已逐项核对；PNG/静态描述来源、native full37 Single I0、A/D独立句子嵌入、GPU调用记账与09:00截止已审查。准备期修正独立入口导入顺序、BF16/FP32噪声摘要和内存KV签名；没有失败GPU尝试。真实新fixture正确性仍待编码后审核。

**批准GPU0仅执行P1两初图编码：最多6次text encoder forward、2次image VAE encode、0视频encode、0denoiser forward、0decode、≤900 GPU秒。** 启动前确认实际空闲、单卡allocated≤44GiB、磁盘≥60GiB。不改冻结代码/config，不自动重试。完成后跑CPU audit_fixtures.py并提交真实新输入哈希和账本。

G1四个首窗、G2续写尚无GPU授权；P1成功不等于生成能力通过。总任务232推理forward/12decode、编码.25+推理1.25GPUh与09:00截止不变。
