# EXP-010核心阶段审核与C7/C8放行

2026-10-11 05:23 HKT。**全程FM8 AA124有限可行性接受，quality PARTIAL；C7/C8尚待验证。** Judge看过C4–C6全部51新增帧，人物与停车场持续可辨，A方向保留；flow依次+.8356/+.7319/+.6603。边界视角/姿态变化、场景几何跳变和腿部透明残影保留，不追加画质调参。

独立CPU审计PASS：36项来源、已保存端点/实际noise/prompt/history、旧73RGB逐值不变、各段MP4完整解码/24fps/PTS，以及真实C6 cache tensors所有50层精确indices核对。C6已真实淘汰C1，只留C2–C6（indices1–5），300层级commits、14,164,800,000 bytes。当前证明首次淘汰写入正确，使用淘汰后历史生成新块的能力仍待C7/C8。

实际28forward=24sampling+4commit、3VAE、0训练，565.081841秒=.156967178GPUh，allocated峰26.45499GiB。独立机器审计见judge/core_audit.json，逐帧观察见judge/visual_notes.json。

现在批准GPU0 A继续/D切换的C7/C8，新增最多34forward=32sampling+2commit、4VAE。总任务62forward/7VAE/.75GPUh不变、单卡44GiB、09:00截止。共享全FM8 AA124与C6 KV，两路同C7初噪声；C8用各自C7。C7读取1–5，C7提交后C8读取2–6；检查实际14,164,800,000 bytes容量、历史RGB不改写与晚切换反应。每块看新增画面，持续结构崩溃则该分支停止，不自动重试、不加C9。

与已有30步SW-G完整158视频作各自历史的方案对比，不称同KV单状态消融或公平E2E速度排名。来源冻结不变，CORE_JUDGE_REVIEW.json与GPU_AUTHORIZATION_BRANCH.json已写。
