# EXP-009 Judge C2审核与C3放行

C2完整来源/执行审计PASS，22forward/4VAE/0update，447.449484秒=.124291523GPUh，allocated峰25.96971GiB。两个模型各自KV、28项来源、匹配history/noise/prompt、历史RGB不变、56帧24fps/PTS完整解码均通过。证据见c2_audit.json；不把工程通过视为生成质量通过。

Judge看过四段全部新增17帧与AF4 AD原分辨率末帧。FM4 AA/AD动作方向可辨，flow+.706/−.190，人物/停车场可用，残影和边界变化PARTIAL。AF4 AA仍可用、flow+.489；AF4 AD flow+.162，D响应弱/含混、人体透明与场景纹理扭曲更明显。没有出现DMD式全画面噪声或主体彻底消失。AF4当前没有整体优于FM4的证据。

按原有限计划批准两个模型AA/AD C3，各用自己生成的C2与KV，新增最多16forward/4VAE/0update，任务累计仍38forward/8VAE/.6GPUh，GPU0、44GiB、09:00截止。只完成这一轮累积表现对比，不追加2NFE或训练。来源SHA不变；主marker及独立C3 marker已写。完成后正式收口，普通缺陷PARTIAL，不因单个proxy轻微反号就否定所有可用证据。
