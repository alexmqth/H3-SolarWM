# DMD8评估入口预审

2026-10-11，Judge。已审查run_eval.py相对已验收AF3 runner的差异，以及FM8/AF8/DMD8三列CPU视频渲染脚本。候选checkpoint改为EXP-008/v2最后完整cycle；评估协议继续使用共同FM8首窗、自己权重prefill KV、相同seed13噪声/动作/native8sigma，C2同历史、C3各自历史。没有引入额外time/prefix/action条件改动。

独立运行CPU preflight，真实返回WAITING_FOR_DMD_TRAIN_CHECKPOINT、0GPU调用。当前训练尚未终止，未冻结最终checkpoint来源，也没有EVAL_GPU_AUTHORIZATION.json。只有训练最终审核后才能放行预定义35forward/4VAE/.35GPUh，不凭准备代码自动推理。

发布前要求最终训练budget已关闭、唯一最后完整checkpoint及四文件receipt可追溯，评估source manifest与独立marker绑定。三列视频沿用原片，不裁去画面或遮挡内容，标注共同首窗、C2 matched history、C3 own history及实际final cycle。不能由预检或finite forward声称DMD生成能力通过。
