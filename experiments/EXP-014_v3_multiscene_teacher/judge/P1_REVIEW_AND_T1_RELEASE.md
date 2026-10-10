# EXP-014 P1验收与T1放行

2026-10-11，Judge。四图native编码通过独立CPU验收：实际12text/4image encode、110.213938886 GPU秒、allocated峰40.577766GiB，0denoiser/0decode/0训练，无重试。GPU进程已正常结束。

核对四fixture真实SHA、PNG/静态caption/train来源、Single I0 390rows、CPU原生video/audio噪声逐值重建、A/D仅动作prompt行不同，以及stop12/17裁剪保留原full37 Global坐标。见[P1独立审计](P1_INDEPENDENT_AUDIT.json)。

现仅放行第一图s0_43866101在GPU0完成A首39帧与同C1历史AA/AD C2到56：90sampling+1clean commit、3decode、1350GPU秒。生成内容仍待视觉与动作验收；余三图T2没有授权。见[T1 marker](T1_APPROVED.json)。
