# Code map

- causal/h3_cached.py：chunk-wise attention、persistent raw KV、clean commit、RGB/image anchor 和 action feedback。
- causal/benchmark.py：original/recompute/cached causal inference 与性能/视频指标。
- causal/pretrained_lora.py：实验 adapter 的安装、保存和加载。
- causal/train_online_selfrollout.py：generated-history per-sigma teacher replay、paired action diagnostic。
- causal/stage2_lite_dmd.py：shared-backbone student/critic/teacher Stage2-lite chain。
- causal/evaluate_action_control.py：signed horizontal optical-flow proxy。
- causal/probe_action_geometry.py、probe_action_routing.py：teacher/causal score geometry 和 routing audit。
- abot/：最小 H3 action preset 和 inference loader。
- diffsynth_h3_action.patch：H3-World action-directed attention patch。
- diffsynth_causal.patch：causal/cache 相关 DiffSynth 变更。

所有 .py 文件均为源代码；提交时已排除 __pycache__、latent、conditioning 和运行日志。
