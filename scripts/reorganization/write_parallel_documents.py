"""Current presentation: two parallel repairs, then planned efficient unification."""
from pathlib import Path
import json
import os
import re

SUB=Path(__file__).resolve().parents[2]
AUDIT=SUB/'archive/taxonomy_v2a_v2b_20261010'


def put(rel,text):
    p=SUB/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.rstrip()+'\n')


TABLE='''| 版本 | Action fidelity | Visual stability | 已验证范围 | Persistent KV | 采样 | 新增训练 | 当前结论 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D方向正控 | 124f基本完整 | 124f及已有长片参考 | 否 | 30整段；另存50步 | 无，released LoRA | Reference，非GT |
| V1 Native causal | 显著下降 | 本代表停滞、过亮/背景退化；其他早期协议有重影 | 124f工程rollout | 是，CPU raw video KV | 8/chunk | 本代表无 | 工程可行，联合质量失败 |
| V2a RGB-Anchor | A/D方向失败 | **124f人物结构相对稳定**；20s失败 | 124f视觉证据；243/481f负结果 | 是，clean commit的历史hidden KV | 8/chunk | visual adapter +已训action residual | Longer-horizon Visual Stability Demonstrated（仅124f scope） |
| V2b Same-σ local bidir | 两history的第二块A/D方向正确 | **56f第二块人物基本完整** | 12+5latent、两块56f | **否**，历史/当前局部双向重算 | 30/chunk | **无**，从Original出发 | Local Visual and Action Fidelity Demonstrated |
| V3 Efficient causal | 目标：保留 | 目标：保留并扩大范围 | **尚未实现** | 目标：strict causal + KV | 先可信30步，后AnyFlow | 待定 | Planned Unification，无模型/视频 |
'''


COMPARE='''| 维度 | V2a：RGB-Anchor Causal | V2b：Same-σ History / Local Bidir |
|---|---|---|
| 共同问题 | 因果rollout中生成历史/条件与原生模型行为不匹配 | 同一研究问题的另一条路线 |
| 核心思路 | 修复图像条件协议，配合视觉适配 | 恢复原生条件和尽可能接近原生的局部联合去噪 |
| 图像条件 | RGB-consistent dual anchor | Single I0 |
| 历史 | 自己生成的clean endpoint经clean commit写入KV | 自己生成的history按当前σ临时加噪，每步重算 |
| Attention | Strict chunk-causal video attention | 已可见history与当前video局部双向；未知未来移除 |
| Persistent hidden KV | 支持，CPU raw video K/V | 不支持 |
| 采样 | 8steps/chunk | 30steps/chunk |
| 新增训练 | visual QKV + endpoint/boundary/replay；使用已训action residual | 无；Original + released action LoRA |
| 视觉证据 | 124f人物/场景相对完整；20秒崩坏 | 56f中第二块人物结构基本完整 |
| 动作证据 | A/D方向门槛失败 | AA/AD/DA/DD第二块方向正确 |
| 主要不足 | 动作控制；更长时程也退化 | 历史重算成本、效率与长时可靠性未验证 |
'''


def main():
    migration=json.loads((AUDIT/'migration.json').read_text());pathmap=migration['file_path_map']
    # Correct planned-directory references after ID remapping and remove sequential language.
    for base in ['report','mainline','branches']:
        for p in (SUB/base).rglob('*.md'):
            t=p.read_text().replace('V2b_efficient_causal_planned','V3_efficient_causal_planned')
            t=t.replace('V2a→V2b','V2a vs V2b（并列路线）')
            p.write_text(t)
    for f in ['README.md','REPORT.md','REORGANIZATION_SUMMARY.md']:
        p=SUB/f;p.write_text(p.read_text().replace('V2b_efficient_causal_planned','V3_efficient_causal_planned').replace('V2a→V2b','V2a vs V2b（并列路线）'))
    # Current manifests resolve to current copies; historical source IDs remain explicit.
    def remap(obj):
        if isinstance(obj,str):return pathmap.get(obj,obj)
        if isinstance(obj,list):return [remap(v) for v in obj]
        if isinstance(obj,dict):return {k:remap(v) for k,v in obj.items()}
        return obj
    for p in [*SUB.glob('mainline/*/manifest.json'),*SUB.glob('report/V*/PROVENANCE.json')]:
        obj=remap(json.loads(p.read_text()))
        if 'V2a_' in str(p):obj['display_version']='V2a'
        if 'V2b_' in str(p):obj['display_version']='V2b'
        p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    p=SUB/'branches/EXPERIMENT_INDEX.json';t=p.read_text()
    t=re.sub(r'V([234])(?![0-9ab])',lambda m:{'2':'V2a','3':'V2b','4':'V3'}[m.group(1)],t)
    p.write_text(t)
    for p in SUB.glob('branches/*/*/manifest.json'):
        p.write_text(re.sub(r'V([234])(?![0-9ab])',lambda m:{'2':'V2a','3':'V2b','4':'V3'}[m.group(1)],p.read_text()))
    put('report/roadmap.md','''# 研究主线：V2a / V2b是并列探索，V3才是未来统一

V0建立原始能力，V1实现基本因果化并暴露动作/视觉退化。V2a和V2b都针对生成历史/条件协议与原生模型不匹配的问题，探索不同解法；**二者不是先后升级关系，也没有checkpoint继承关系**。

```mermaid
flowchart TD
    V0["V0 Original H3-World — Bidirectional Baseline"] --> V1["V1 Native Chunk-Causal — KV / visual & action degradation"]
    V1 --> V2a["V2a RGB-Anchor — Visual Stability Repair"]
    V1 --> V2b["V2b Same-σ / Local Bidir — Local Visual & Action Recovery"]
    V0 -. "Restore native H3 protocol / Original weights" .-> V2b
    V2a -. "visual repair + history reuse insights" .-> V3["V3 Efficient Causal H3-World — Planned Unification"]
    V2b -. "local action / visual evidence" .-> V3
    V3 --> AF["Future: AnyFlow Acceleration"]
    AF --> DMD["Future: On-policy DMD"]
    A["Branch A: causal mechanism diagnostics"] -.-> V3
    B["Branch B: causal adaptation / action recovery"] -.-> V2a
    C["Branch C: preliminary AnyFlow / DMD"] -. "past explorations, not V2b completion" .-> AF
    classDef planned fill:#e5e7eb,stroke:#6b7280,color:#111827
    class V3,AF,DMD planned
```

图中V1到两支表示研究问题的分叉；V0到V2b说明实际权重/协议来源。V2a与V2b之间没有继承箭头。向未来V3的虚线表示需要吸收的研究证据，不代表把两个checkpoint拼接就能成功。

'''+COMPARE+'''
## 证据范围必须分开

**V2a — Longer-horizon Visual Stability Demonstrated：仅指124帧的相对视觉证据。** A/D仍失败，同checkpoint20秒视频严重退化，不能说已经解决长期崩坏。RGB anchor、visual adapter、endpoint/boundary监督与routing共同变化，不能把全部提升只归因anchor。

**V2b — Local Visual and Action Fidelity Demonstrated：仅指56帧中的第二块。** 原生Single I0、native time、Same-σ和C12→5/T2联合协议在自身history下有效；没有第三/第四块、124f或跨场景结果，也不支持persistent hidden KV。

## 未来V3：需要训练/设计一个可信的严格因果骨干

目标为 **Efficient Causal Generation + Visual Stability + Action Fidelity**。研究上希望吸收V2a的历史复用和视觉修复经验，以及V2b的局部动作/结构能力；但当前没有证据证明简单合并anchor、adapter或checkpoint即可实现。

真正缺少的是：在**不依赖history与current video双向重算**时，仍有正确动作条件能力的causal模型。应先验证strict causal30的局部动作与结构，再评估persistent KV与长时生成；之后AnyFlow学习少步，on-policy DMD处理自身rollout分布。

此前AnyFlow/DMD是旁路先行探索，不是V2a/V2b已经完成相应阶段。[两支能力对比](00_comparison_gallery/V2a_vs_V2b.mp4) · [下一步验收](02_next_steps/README.md) · [首页](README.md)
''')
    put('report/README.md','''# H3-World × SolarWM：一分钟汇报导航

**目标：** 将SolarWM的因果分块、KV cache与少步方法迁入H3-World，同时保留action control与画面连续性，最终改善长视频效率。

**当前结论：** V1证明strict causal/KV工程可行，但损害动作/画面。V2a和V2b是针对同一问题的**两条并列修复路线**：V2a联合RGB视觉适配，在124帧改善结构但动作失败；V2b从Original恢复原生条件，Same-σ加局部双向重算，在56帧第二块取得局部动作与结构正结果。没有一个版本同时完成动作、长期稳定和整体加速；未来V3负责统一这些能力。

**建议播放顺序：**

1. [V0 vs V1：直接因果化的代价](00_comparison_gallery/V1_vs_original.mp4)。
2. [V1 vs V2a：RGB联合视觉修复](00_comparison_gallery/V2a_vs_V1.mp4)。
3. [V1 vs V2b：恢复原生条件与局部联合重算](00_comparison_gallery/V2b_vs_V1.mp4)。
4. [V2a vs V2b：并列路线的能力取舍](00_comparison_gallery/V2a_vs_V2b.mp4)，采样、history和KV不同，非单变量消融。
5. [V2b四路径56f](00_comparison_gallery/V2b_four_paths_56.mp4)，RGB39高亮第二块；[V2a完整20秒失败片](V2a_rgb_anchor/videos/V2a_long20s_failure.mp4)保留后段。

'''+TABLE+'''
## 按需展开

- [V0](V0_original/README.md) / [V1](V1_native_causal/README.md) / [V2a](V2a_rgb_anchor/README.md) / [V2b](V2b_same_sigma_local_bidir/README.md)：统一十项说明、代码快照、真实原片。
- [22条比较/参考画廊](00_comparison_gallery/README.md) / [本地浏览器演示页](index.html)。
- [并列路线图与维度对照](roadmap.md) / [研究分支](01_research_branches_summary/README.md) / [5分钟讲稿](TALK_5MIN.md)。
- [未来V3与验收](02_next_steps/README.md) / [历史指标](METRICS.md) / [公平性与缺失](COMPARISON_PROTOCOL.md)。

当前局部动作＋视觉最好证据来自V2b；较长的124f视觉修复证据来自V2a。**不以56f局部结果覆盖124f/20s结论，也不把V2a称为已实现端到端加速。**

本目录可独立复制汇报：MP4均为真实文件，核心code供阅读，不含33B权重。当前标签与文件名均使用V2a/V2b；旧V2/V3编号片保留在完整仓库archive内。本次只做CPU素材整理，不训练或进行模型推理。
''')
    put('mainline/README.md','# Mainline · 原始能力、因果化与两条并列修复路线\n\nV0 → V1 → {V2a, V2b} → V3 Planned → AnyFlow → On-policy DMD。这是研究路线；V2b权重从Original出发，V2a与V2b之间没有checkpoint继承。\n\n'+TABLE+'''
- [V0 Original Bidirectional](V0_original_bidirectional/README.md)
- [V1 Native Chunk-Causal](V1_native_chunk_causal/README.md)
- [V2a RGB-Anchor Causal](V2a_rgb_anchor_causal/README.md)
- [V2b Same-σ History / Local Bidir](V2b_same_sigma_local_bidir/README.md)
- [V3 Efficient Causal — Planned](V3_efficient_causal_planned/README.md)

[汇报包](../report/README.md) · [路线图](../report/roadmap.md) · [研究分支](../branches/README.md)
''')
    put('mainline/V3_efficient_causal_planned/README.md','''# V3 · Efficient Causal H3-World — Planned Unification

1. **Version Name / Research Objective**：统一 Efficient Causal Generation + Visual Stability + Action Fidelity。
2. **Parent Version / Baseline**：V1问题、V2a视觉/缓存经验、V2b局部动作/结构正控；没有直接继承一个已合格的checkpoint。
3. **Main Changes**：研究strict chunk-causal下保留原动作信息流，明确public prefix、历史时间与action routing，配合必要causal adaptation和正确persistent KV。
4. **Model and Inference Configuration**：尚未确定。先30step局部可信，再历史复用与多块，之后AnyFlow、on-policy DMD。
5. **Representative Videos**：**NOT_YET_VALIDATED，无V3视频**。
6. **Quantitative Results**：无V3模型性能结果。
7. **What Was Improved**：已有两支证据明确能力取舍和受控诊断方向，不是V3已改善。
8. **What Still Failed**：strict causal + KV尚未通过动作、视觉与长时效率联合验收。
9. **Lessons Learned**：不能简单拼接两个checkpoint/anchor；核心是让模型在不依赖历史与当前双向重算时仍有正确动作条件能力。
10. **Source / Checkpoint / References**：无V3 checkpoint。[V2a与V2b路线](../../report/roadmap.md) · [验收条件](../../report/02_next_steps/README.md)。

V2a的124f视觉证据和V2b的56f局部动作证据不能直接相加成一个成功模型。本次不启动新实验。
''')
    for d,id_ in [('report/V2a_rgb_anchor','V2a'),('report/V2b_same_sigma_local_bidir','V2b')]:
        p=SUB/d/'README.md';t=p.read_text()
        marker='## 2. Parent Version / Baseline\n\n'
        start=t.index(marker)+len(marker);end=t.index('\n## 3.',start)
        parent=('针对V1暴露的生成历史/条件不匹配问题的一条并列修复路线；使用RGB visual QKV与已有fixed-mix action residual。与V2b没有顺承/权重继承关系。'
                if id_=='V2a' else '**从Original H3 + released action LoRA恢复原生条件**，针对V1暴露的问题探索另一条并列路线。没有使用V2a visual adapter，不是V2a checkpoint续训。')
        t=t[:start]+parent+'\n'+t[end:]
        if id_=='V2a':
            t=t.replace('## 5. Representative Videos\n\n','## 5. Representative Videos\n\n- [V2a vs V2b：并列能力比较，非单变量消融](videos/V2a_vs_V2b.mp4)\n')
            t=t.replace('## 7. What Was Improved\n\n','## 7. What Was Improved\n\n**Longer-horizon Visual Stability Demonstrated — 仅124f的相对视觉证据，非无限长稳定。**\n\n')
        else:
            t=t.replace('## 5. Representative Videos\n\n','## 5. Representative Videos\n\n- [V1 vs V2b：共同前56f，协议不同](videos/V2b_vs_V1.mp4)\n')
            t=t.replace('## 7. What Was Improved\n\n','## 7. What Was Improved\n\n**Local Visual and Action Fidelity Demonstrated — 仅56f第二块，未验证长期稳定。**\n\n')
            t=t.replace('V2a vs V2b（并列路线），跨协议研究对比','V2a vs V2b，并列能力比较，非升级关系')
        p.write_text(t)
        mainname={'V2a':'V2a_rgb_anchor_causal','V2b':'V2b_same_sigma_local_bidir'}[id_]
        mp=SUB/'mainline'/mainname/'README.md'
        old=mp.read_text();tail=old[old.index('## 完整证据与边界'):]
        # Rebase report links while retaining the original detailed mainline references.
        head=re.sub(r'\]\(([^)]+)\)',lambda m:']('+os.path.relpath(p.parent/m.group(1),mp.parent)+')',t)
        mp.write_text(head+'\n'+tail)
    # Main root entry and concise report remain useful, but no linear V2a -> V2b story.
    p=SUB/'README.md';t=p.read_text();start=t.index('## 版本比较');end=t.index('## 目录',start)
    t=t[:start]+'## 版本比较\n\n'+TABLE+'\nV2a与V2b共同研究生成历史/条件不匹配的修复，没有顺承关系。V2a为RGB联合视觉适配；V2b从Original恢复Single I0/native条件，Same-σ与局部双向重算，仅验证第二块。未来V3才是严格因果、KV、视觉和动作的统一目标。\n\n'+t[end:]
    t=t.replace('19条画廊','22条画廊').replace('V0 Original → V1 native → V2a RGB → V2b Same-σ → V3 Planned','V0 → V1 → {V2a RGB, V2b Same-σ} → V3 Planned')
    t=t.replace('最新V2b从Original重新出发','并列路线V2b从Original重新出发')
    p.write_text(t)
    p=SUB/'REPORT.md';t=p.read_text().replace('新V2b回到Original权重','另一条并列路线V2b回到Original权重')
    t=t.replace('## 原型与研究路线\n\n','## 原型与研究路线\n\nV2a与V2b是同一问题的两种修复探索，无先后升级关系。未来V3目标是在strict causal、persistent KV下统一视觉与动作，不是简单拼接两套checkpoint。\n\n')
    t=t.replace('- [V2b四路径，56f]', '- [V1 vs V2b，前56f](report/00_comparison_gallery/V2b_vs_V1.mp4)：恢复原生协议的并列路线，非单因素消融。\n- [V2b四路径，56f]')
    p.write_text(t)
    put('report/02_next_steps/README.md','''# V3 Planned：统一两条路线的能力，随后才是AnyFlow与DMD

**目标：Efficient Causal Generation + Visual Stability + Action Fidelity。** V2a提供CPU KV/视觉修复经验，V2b提供局部动作与结构正控；它们不是可直接拼接即合格的两个组件。

真正缺少的是经过适当causal adaptation的模型：不依赖history/current video双向重算，仍有正确动作条件能力。训练目标和拓扑选择须基于受控证据，不能因两支各有优点就直接宣称统一可行。

| 次序 | 需要回答的问题 | Go条件 / 当前边界 |
|---|---|---|
| 局部能力 | strict chunk-causal是否保留动作与结构 | 足够30steps、同history/noise与当前A/D；视频方向/结构共同评审，不单看cosine |
| 缓存 | 实际cache与匹配时间/输入的重算是否等价 | 逐层K/V、RoPE、velocity跨sigma；区别matched rebuild与sigma0 clean commit；已有诊断wrapper证明受控等价，非默认路径自动通过 |
| 多块 | 能否在自己的history继续且不崩 | 第三/第四块、多场景，再124f；目前V2b只到第二块56f |
| 效率 | 历史复用是否带来端到端收益 | 统一硬件/offload、warmup、多次均值、真实首屏与每块计时；V2a也没有整体加速证明 |
| AnyFlow | 可否降低每chunk采样 | 在已可信causal30上比较4/8步，finite/diagonal与teacher endpoint双指标及完整视频 |
| On-policy DMD | 能否改善生成历史分布偏移 | 真self-rollout、frozen teacher、trainable fake score与正确生成梯度；已有lite质量失败/工程准备不能当完整Stage2 |

V2a的124f视觉证据不覆盖20秒；V2b的56f局部动作与视觉证据不覆盖124f。V3无checkpoint和视频。本次只是重整研究路线，没有启动上述实验。

[路线图](../roadmap.md) · [汇报导航](../README.md)
''')
    put('report/TALK_5MIN.md','''# 5分钟答辩：两条并列路线的取舍与统一目标

**0:00–0:45：目标与因果化代价。** 播放[V0 vs V1](00_comparison_gallery/V1_vs_original.mp4)。我们要把SolarWM的causal/KV/少步思路迁到H3-World。V1工程可执行，但A/D近乎停滞；工程实现与能力验收分开。

**0:45–1:35：为什么action信息流重要。** Original是Single-Egress，多层video传播负责间接传动作；causal action prefix增加past-action直读，来自fresh prefix而非缓存action。CPU cache保存raw video K/V，clean commit和滑窗。受控数值P0等价不表示动作好，无persistent KV的strict图已经失配。

**1:35–2:30：同一问题的两条修复路线。** 播放[V2a vs V2b](00_comparison_gallery/V2a_vs_V2b.mp4)。V2a修复RGB图像条件并训练visual adapter，支持strict causal/KV，124f结构改善，A/D失败；V2b从Original恢复Single I0/native time，用Same-σ和局部双向重算，无新增训练也无persistent KV。两者不是升级关系，步数/history/cache也不匹配，比较的是能力取舍。

**2:30–3:25：正结果与失败边界。** 播放[V2b四路径](00_comparison_gallery/V2b_four_paths_56.mp4)，RGB39进入第二块：自身history，无GT reset，AA/AD/DA/DD局部方向和人物结构成立。随后展示[V2a 20秒失败](V2a_rgb_anchor/videos/V2a_long20s_failure.mp4)后段；V2a只证明124f视觉相对稳定，V2b只证明56f局部，不能说两者都解决长期崩坏。

**3:25–4:15：少步与训练探索没有神奇修复。** 8/chunk×8=64 noisy+8commits，对照Original30次长序列forward，不能算30/8加速。RGB decode/re-encode、CPU offload和重算成本不同。ordinary FM、真实ABot、AnyFlow16/64/128/136和DMD-lite都保留负结果；不是完整Stage1/2成功。

**4:15–5:00：未来V3的技术判断。** V3要统一高效严格因果生成、视觉稳定和动作保真，但不能直接拼接两个checkpoint。需要一个不依赖history/current双向重算仍有正确动作能力的causal backbone，再做AnyFlow和on-policy DMD。目前没有V3模型/视频，也没有统一硬件重复均值的效率结论。

[导航](README.md) · [维度对照与路线](roadmap.md)
''')
    # Update local player labels/targets: same UI, parallel meaning.
    p=SUB/'report/index.html';t=p.read_text().replace('V2b_efficient_causal_planned','V3_efficient_causal_planned')
    t=t.replace('V2a→V2b 跨协议','V2a vs V2b 并列能力比较')
    t=t.replace('V1→V2a','V1 vs V2a')
    t=t.replace('<option value="00_comparison_gallery/V2b_four_paths_56.mp4">','<option value="00_comparison_gallery/V2b_vs_V1.mp4">V1 vs V2b</option><option value="00_comparison_gallery/V2b_four_paths_56.mp4">')
    t=t.replace('V0动作正控 → V1工程跑通但动作减弱 → V2a联合视觉修复、动作未恢复 → V2b回到Original恢复原生条件并重算历史，只有第二块局部通过 → V3高效化仍待研究。','V0动作正控 → V1基本因果化 → 两条并列路线：V2a联合RGB视觉修复（124f，动作未恢复）；V2b恢复原生条件、Same-σ局部重算（56f第二块动作与结构正控） → 未来V3统一严格因果、缓存、视觉与动作。')
    p.write_text(t)
    # Preserve discarded screen labels and the old auditing chain, with a mapping entry.
    put('archive/obsolete_presentation_v2_v3/README.md','''# 已弃用的演示编号：原V2 / 原V3

这些MP4是2026-10-10整理初版的真实编码结果，字节未变。按用户纠正，新编号为原V2→**V2a**（RGB）、原V3→**V2b**（Same-σ局部双向）；原V4计划→**V3 Planned**。旧片顶部标签不是现行版本，不用于正式展示。

原V2与原V3不是顺承关系；正式[新画廊](../../report/00_comparison_gallery/README.md)已CPU重新制作V2a/V2b标签，并增加V1 vs V2b及两支能力对比。原始实验MP4完全未变。

[逐文件迁移/哈希](../taxonomy_v2a_v2b_20261010/migration.json) · [当前路线](../../report/roadmap.md)
''')
    put('archive/taxonomy_v2a_v2b_20261010/README.md','''# V2a / V2b 并列路线修订

用户纠正：RGB anchor与Same-σ是同一条件/历史问题的两种研究路线，不是顺承版本。将原V2→V2a，原V3→V2b，原V4 Planned→V3 Planned，目录、文档和屏幕标签同步。

- [修订前文件SHA](before_files.json) / [git mv与路径映射](migration.json)。
- [本次来源映射](selected_sources.json)：保留legacy asset ID用于追溯，不与当前display_version混淆。
- [新视频配方](render_specs.json) / [编码与解码结果](render_results.json) / [汇报副本](gallery_copies.json)。
- [完整修订验收](validation.json)：原视频不覆盖、代码/测量不改、链接/解码/便携性。

旧编号MP4在[obsolete_presentation_v2_v3](../obsolete_presentation_v2_v3/README.md)归档；初次整理的原始审计收据保持当时路径，结合本次migration查新位置。本次只有CPU编码/文档操作，无模型训练或推理。

本次可重放的渲染源码：[render_parallel_gallery](../../scripts/reorganization/render_parallel_gallery.py)。迁移/文档脚本为一次性来源记录；现行验收入口为[validate_parallel_taxonomy](../../scripts/reorganization/validate_parallel_taxonomy.py)。旧prepare/finalize脚本描述初次编号，不再用于现行目录。
''')
    # Mainline branch matrix is useful outside the compact report too.
    put('mainline/PARALLEL_REPAIRS.md','# V2a / V2b：并列修复路线\n\n'+COMPARE+'\n二者没有checkpoint继承，亦非单因素消融。V2a仅124f视觉证据，V2b仅56f第二块局部动作/结构。未来V3需真正统一严格因果、历史复用与能力，不是checkpoint拼接。\n\n[能力比较视频](../report/00_comparison_gallery/V2a_vs_V2b.mp4) · [路线图](../report/roadmap.md)\n')
    print('Parallel taxonomy documents written; immutable raw receipts/source code unchanged.')


if __name__=='__main__':main()
