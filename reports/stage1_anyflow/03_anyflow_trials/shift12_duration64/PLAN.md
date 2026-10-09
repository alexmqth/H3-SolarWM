# Shift12 full-history训练量对照：16→32→最多64

仅在GPU1的shift12/16次训练和全部四条generated-history评测完成、所有4/8-step数字gate均未通过之后启动。若16已过数字gate，则停止等待人工视觉复核，不启动无必要训练。

保持同一shift12/16 checkpoint、Adam、CPU/CUDA/logical RNG，先总32更新及A/D8评测；数字gate未过再到最多64及A/D4/8。只改变更新总数，不改architecture、RGB anchor、rank8、FP32、LR、batch4、A/D两条teacher、noise/seed13、chunk5/history5、CPU raw KV、causal action prefix/feedback。

训练shift12，validation/inference保持2.22，与GPU0的shift2.22训练量分支分开。两者都有16/32/64的公共协议评测，用于判断增加训练量是否帮助，以及噪声时间分布的作用；目前不保证会改善。

GPU1/reserve6，每条启动前检查GPU空闲。不打断其它进程。最多新增48次更新与6条39f视频，不扩124f、不做Stage2；数字通过仍需人物/场景视觉完整、匹配FM少步收益、独立seed和动作切换验收。

预计新增训练约2小时以上，另有准备/验证/视频开销；这是按已完成16次训练估算。控制器归档依赖源机器的冻结runtime，通用复现入口见submission/REPRODUCE.md。
