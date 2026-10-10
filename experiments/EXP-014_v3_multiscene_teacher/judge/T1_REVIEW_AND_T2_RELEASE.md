# EXP-014 首图T1验收与余三图T2放行

2026-10-11，Judge。**第一张固定train初图s0_43866101的FM30 C1/AA/AD教师目标有限可行通过，quality PARTIAL。** 全39首窗+两路全部17新帧联系表、首窗38与两路55原分辨率已检查。人物、山坡/道路/树林可辨；AA背景右移、AD左移，运动分叉可辨，保留暗部/软细节/边界变化。没有持续人体瓦解或全景失败。不把教师伪标签叫真实录屏GT，不据此声称训练改善。

[G1](G1_s0_43866101_FM30_audit.json)/[G2](G2_s0_43866101_FM30_audit.json)独立审核PASS：真实输入/30-step native网格、两分支同C1和C2噪声、实际50层index0 cache（6,799,104,000 bytes）、旧39RGB逐值不变及完整MP4解码。模型释放边界仅0.008911GiB残余，不存在两个33B模型叠加。视觉记录见[visual_notes](visual_notes.json)。

实际90sampling+1clean commit=91forward、3decode，532.260703702 GPU秒，allocated峰38.181200GiB，无重试/训练。C1 sampling174.097秒，AA114.337秒，AD111.270秒；commit14.729秒，decode共15.657秒，加载/保存等另计。

现放行[T2](T2_APPROVED.json)剩余三图，分别GPU0/1/2（启动前再确认空闲），独立账本和锁，每图91forward/3decode/1350GPU秒。T2总273forward/9decode、最多4050GPU秒；总任务既定预算不变。每张分别验收，失败保留，不换图/seed、不追加训练。三GPU均在09:00后≤3的项目上限内；本任务仍受08:40新scene/09:00运行闸门。
