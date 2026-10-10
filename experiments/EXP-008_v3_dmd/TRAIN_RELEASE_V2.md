# EXP-008/v2 Judge训练放行

2026-10-11，Judge。v1 pilot工程审核通过后，独立检查v2代码及CPU preflight：29项来源、配套student/fake权重、两个AdamW真实CPU恢复、RNG重放、实际DMD导入与A/D顺序通过。运行时在所有模块安装后恢复全部RNG；teacher C1复用、student/fake每cycle按自身权重重建、fake更新后再次重建，完整8map梯度保留。记录每cycle噪声SHA、缓存断言、base冻结和各卡显存。

**现在批准新增7cycles至累计8：GPU0/2/5，99forward/14backward/14update/0VAE，≤45min wall、保守三卡≤2.25GPUh，每卡allocated≤44GiB，最晚09:00。** 显卡已实时确认空闲。仅保存cycle4/8及预算提前收口的最后完整cycle，异常停止，无自动重试、不增加cycles或配置。marker绑定最终manifest SHA。

任务书taskbook_v2.md保持冻结；视频只有CPU准备授权，须训练报告经Judge审核再放行35forward/4VAE/.35GPUh。训练成功不等于生成能力通过。
