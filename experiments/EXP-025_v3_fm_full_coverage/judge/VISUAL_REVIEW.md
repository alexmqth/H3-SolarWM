# EXP-025 Judge 连续帧审核

2026-10-16 HKT。方式为解码后的连续帧图序列审阅，不是实时播放或自动语义评分。
四条GT-history均审阅RGB35–55（全部17新帧及边界），两条self-history均审阅RGB0–55。共26页，见FINAL_AUDIT.json的visual_review_sheets。

- train_constant GT-history：人物结构可辨，0/256姿态和路径接近，无清晰新增收益。
- train_switch GT-history：尾部迈步和位置有变化，但不足以证明动作切换改善。
- dev_constant GT-history：腿部细节不同，整体前后接近。
- dev_switch GT-history：约f40起两者都有明显视点/构图变化，256未消除。
- dev_constant self-history：两者都能维持单个人物与基本步态，但相对GT有明显运动不足、反复步态；256尾部腿部姿态变化不构成稳定改善。
- dev_switch self-history：两者均从首块开始出现不同于GT的相机移动，人物动作有限；C1/C2边界及后段视点偏移仍在，256与zero总体接近。

结论：visual/action PARTIAL，新增收益未证实。未以轨迹不匹配单独判定动作错误；没有同状态A/D反事实，不能报告方向正确率。每列缩略图用于总体序列审核，不能保证原分辨率下所有细小ghosting均不存在。
