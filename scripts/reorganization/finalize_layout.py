"""Non-destructive navigation refactor; git mv only for historical documents."""
from pathlib import Path
import hashlib
import html
import json
import os
import re
import subprocess
from write_documents import SUB, AUDIT, TABLE, BASE, put, link


def rebase(text,old,new,moves):
    oldp=SUB/old;newp=SUB/new
    def replace(m):
        url=m.group(2)
        if '://' in url or url.startswith(('#','mailto:')):return m.group(0)
        part,sep,anchor=url.partition('#')
        target=Path(os.path.normpath(oldp.parent/part))
        try:rel=str(target.relative_to(SUB))
        except ValueError:return m.group(0)
        rel=moves.get(rel,rel)
        path=os.path.relpath(SUB/rel,newp.parent)+(sep+anchor if sep else '')
        return m.group(1)+path+m.group(3)
    return re.sub(r'(\]\()([^\s)]+)(\))',replace,text)


def main():
    moves={f'docs/archive/{n}':f'archive/historical_docs/{n}' for n in ['NEXT_PLAN.md','PROJECT_PROGRESS.md','SUBMISSION_MANIFEST.md']}
    moves['REPORT.md']='archive/legacy_reports/REPORT_before_reorganization.md'
    actions=[]
    for old,new in moves.items():
        src=SUB/old;dst=SUB/new
        text=src.read_text();dst.parent.mkdir(parents=True,exist_ok=True)
        # Immutable byte-for-byte copy before rebasing historical Markdown links.
        raw=AUDIT/'original_documents'/old
        raw=raw.with_suffix('.txt');raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(src.read_bytes())
        subprocess.run(['git','mv',old,new],cwd=SUB,check=True)
        rendered=rebase(text,old,new,moves)
        if old=='REPORT.md':
            rendered='> 历史报告快照：其中“当前”指当时协议；当前版本结论以[新汇报](../../report/README.md)为准。这里保留原结论与限制，不将其升级为V3或Stage2成功。\n\n'+rendered
        dst.write_text(rendered)
        put(old,'# 历史材料已归档\n\n'+link(old,new,'打开原文归档')+'。这是兼容旧路径的入口；当前研究路线见'+link(old,'report/README.md','新版汇报')+'。\n')
        actions.append(dict(operation='git mv + compatibility stub',source=old,target=new,
                            before_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            transformed='Markdown relative links rebased; original bytes separately preserved',
                            original_bytes=str(raw.relative_to(SUB))))
    put('docs/archive/README.md','# 旧归档路径兼容入口\n\n历史文档已移动到[archive/historical_docs](../../archive/historical_docs/README.md)，三个旧文件名保留跳转页。\n')
    put('archive/historical_docs/README.md','''# 历史进度与计划

以下文档保持当时的研究状态，不用于覆盖当前版本结论。原始字节另保存在整理审计中；这里仅重定位Markdown链接。

- [旧Next Plan](NEXT_PLAN.md)
- [旧项目进度流水账](PROJECT_PROGRESS.md)
- [旧Submission文件清单](SUBMISSION_MANIFEST.md)：清单中的哈希和路径描述原日期，未重写成新测量。

[当前汇报](../../report/README.md) · [归档导航](../README.md)
''')
    # Root README is preserved in raw form and a readable rebased historical copy.
    old=(SUB/'README.md').read_text()
    put('archive/reorganization_20261010/original_documents/README.txt',old)
    put('archive/legacy_reports/README_before_reorganization.md',
        '> 历史导航快照；新入口为[report/README](../../report/README.md)。\n\n'+rebase(old,'README.md','archive/legacy_reports/README_before_reorganization.md',moves))
    actions.append(dict(operation='archive prior README + replace navigation',source='README.md',
                        target='archive/legacy_reports/README_before_reorganization.md',
                        original_bytes='archive/reorganization_20261010/original_documents/README.txt'))
    put('archive/legacy_reports/README.md','''# 整理前的报告与导航

- [整理前REPORT](REPORT_before_reorganization.md)
- [整理前README与实验地图](README_before_reorganization.md)

新总览为[mainline](../../mainline/README.md)、[branches](../../branches/README.md)、[report](../../report/README.md)。旧报告包含当时命名与讨论；新版本映射以新总览为准，未把原始负结果删除。
''')
    put('README.md','''# H3-World × SolarWM：因果化、动作信息流与少步探索

**研究目标：** 将SolarWM的causal chunk、KV cache与少步方法迁入H3-World，检查长视频效率、画面连续性和action control能否同时保留。

**当前判断：工程迁移成立，完整能力目标尚未完成。** strict causal/KV能生成124帧；RGB联合修复改善124帧结构但A/D失败、20秒崩坏；最新V3从Original重新出发，Same-σ + C12→5 + T2联合重算在自身history后的第二块取得局部动作与结构正结果。V3没有persistent hidden KV，不代表完整124帧、AnyFlow或Stage2已经成功。

## 一分钟入口

| 使用场景 | 入口 |
|---|---|
| 组会直接展示 | **[report/README.md](report/README.md)** / [浏览器本地演示页](report/index.html) |
| 连续播放全部对比 | [19条画廊](report/00_comparison_gallery/README.md) |
| 当前局部正结果 | [V3四路径56f](report/00_comparison_gallery/V3_four_paths_56.mp4) |
| 核心模型/协议演进 | [mainline V0–V4](mainline/README.md) |
| 机制、训练、AnyFlow/DMD细节 | [Research Branches A/B/C](branches/README.md) |
| 面试题回答与报告 | [INTERVIEW_ANSWER](INTERVIEW_ANSWER.md) / [REPORT](REPORT.md) |
| 运行环境、权重、adapter | [REPRODUCE](REPRODUCE.md) / [checkpoints](checkpoints/README.md) |
| 本次移动/复制/视频/验收 | [REORGANIZATION_SUMMARY](REORGANIZATION_SUMMARY.md) |

## 版本比较

'''+TABLE+'''
V1主片明确选**零新增adapter、latent dual、seed13、8steps/chunk**的native配置；最早seed2/4-step另存，fixed-mix训练归Branch B。V2视觉结果来自RGB anchor + visual adapter +监督等联合协议，不能只归因anchor。V3从Original恢复原生条件，不继承V2 checkpoint；其视频比较是跨协议研究对比。

## 目录

```text
mainline/             V0 Original → V1 native → V2 RGB → V3 Same-σ → V4 Planned
branches/             A机制诊断 / B因果适配与动作恢复 / C AnyFlow与DMD探索
report/               可独立复制的简洁汇报：核心源码、真实视频、版本说明、5分钟讲稿
archive/              历史文档、旧导航、整理前索引、hash与视频制作/验收收据
code/ checkpoints/    现有运行实现与小adapter；保持原路径
experiments/ reports/  原始实验说明、冻结源码、指标/日志/视频；按新分支索引，原路径保留
meeting/              原会议包与旧主片；作为历史证据保留，最新展示使用report/
docs/ scripts/ tests/  原长文档、运行脚本与测试；未修改生产模型实现
```

旧`reports/stage1_anyflow`是历史文件夹名，含FM、诊断等多个目标，不等于全是AnyFlow。旧结果与measurement不搬乱；新分支逐项建立链接和manifest，减少对原运行脚本的路径破坏。原始大checkpoint和latent仍在外部outputs，由来源清单关联。

**计数口径：** 8steps/chunk×8chunks=64 noisy forwards + 8 commits，不是全视频8次。KV复用不等于AnyFlow少步训练，DMD-lite不等于完整Stage2。耗时为共享硬件单次记录，无warmup均值或公平speedup结论；flow/cosine/MAD均不能单独代替动作与画质评审。

底座与released LoRA不在仓库内，见复现文档。本次只整理现有证据并CPU编码对比，无训练、33B推理、AnyFlow或DMD运行。
''')
    put('REPORT.md','''# 面试交付报告：因果化可行，但动作、长时稳定与效率尚未同时成立

**摘要。** 在真实H3-World权重上，我们实现了chunk-wise causal attention、persistent raw video KV、clean commit、滑窗与CPU offload，完成124帧及更长rollout。直接因果化损害动作信息流；RGB联合视觉修复改善124帧结构，却没有恢复A/D，20秒仍失败。新V3回到Original权重，在原生条件、Same-σ与C12→5可见窗口联合重算下，第二块动作与人物结构局部通过。完整长期控制与端到端加速尚未证明。

## 方法理解与实现

SolarWM仓库Stage0.5使用普通FM适配双向teacher；Stage1把causal teacher forcing和AnyFlow有限区间映射结合，建立少步能力；Stage2在student自己的generated trajectory上用frozen teacher与trainable fake score提供DMD方向。H3-World额外有Single-Egress的Action–Video关系，直接收紧attention可能改变已训练的多层传播，不能只保留action接口就认为控制保真。

核心实现：[h3_cached.py](code/causal/h3_cached.py)、[原动作patch](code/diffsynth_h3_action.patch)、[AnyFlow](code/causal/anyflow.py)、[DMD-lite](code/causal/stage2_lite_dmd.py)。最新V3实际冻结实现见[代码快照](report/V3_same_sigma/code/README.md)。

## 原型与研究路线

[V0–V4详细版本](mainline/README.md)区分零训练native、ordinary FM/visual/action适配、AnyFlow和DMD。V3不使用V2 visual adapter，T2在每sigma重算可见video/prefix hidden；它只保证跨窗口不读未知未来，不是strict chunk-causal attention，更不复用persistent hidden KV。

## Demo、结果与测量

- [Original vs V1，124f](report/00_comparison_gallery/V1_vs_original.mp4)：A/D几乎静止，验证直接因果化的代价。
- [V1 vs V2，124f](report/00_comparison_gallery/V2_vs_V1.mp4)：RGB+visual adapter等联合协议改变结构和运动；A仍方向错。
- [V3四路径，56f](report/00_comparison_gallery/V3_four_paths_56.mp4)：39RGB自身首段后，当前A/D方向和人物结构局部通过。
- [V2 vs V3，共同前56f](report/00_comparison_gallery/V3_vs_V2.mp4)：跨协议比较，不能将改善只归因Same-σ。
- [V2完整20秒失败片](report/V2_rgb_anchor/videos/V2_long20s_failure.mp4)：不能宣称长期稳定已解决。

[指标表](report/METRICS.md)保留E2E/sampling范围、峰值allocated、CPU KV、forward和flow；[公平性说明](report/COMPARISON_PROTOCOL.md)列明共同项与差异。当前不提供虚构FVD/LPIPS/VBench，也不把MAD当质量。历史单次硬件配置不同，不按30/8步数计算速度比。

## 失败分析与技术判断

AnyFlow16/64/128/136、真实ABot FM48、E2 FM+action及Stage2-lite均有实际工程/训练证据，未过对应联合质量门槛。[三个研究分支](branches/README.md)保留负结果与视频；不将其附会为V3完成Stage1/Stage2。

最新[固定状态KV/路由审计](reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)先证明受控数值下cache/recompute逐层误差0，再隔离public prefix、strict video图、commit时间冻结和past-action直接访问。无persistent KV的R2动作差分对Original已失配，说明不能简单归因缓存bug；cosine仍不是视频质量结论。

## 局限与下一步

V3只有单场景/seed、两块56帧，缺第三/第四块与跨场景；V4应先将局部能力迁回strict causal/KV，再AnyFlow减少步数，后on-policy DMD处理生成历史分布。没有在本次整理中启动这些实验。[5分钟讲稿](report/TALK_5MIN.md)、[roadmap](report/roadmap.md)、[完整旧报告归档](archive/legacy_reports/REPORT_before_reorganization.md)。
''')
    banners={
      'meeting/README.md':'> **2026-10-10：新的精简汇报入口为[report](../report/README.md)，版本比较见[画廊](../report/00_comparison_gallery/README.md)。** 本目录保留原会议包及指标；旧主片属于V2 RGB checkpoint，不是最新V3/E2/AnyFlow。\n\n',
      'reports/stage1_anyflow/README.md':'> **新版导航：** [A机制诊断](../../branches/A_causal_diagnostics/README.md) / [B因果适配](../../branches/B_causal_adaptation/README.md) / [C AnyFlow与DMD](../../branches/C_anyflow_dmd/README.md)。本目录保留原证据路径。下文冻结/no-go指当时E2；后来的V3仅在第二块局部通过，不代表训练恢复。\n\n',
      'experiments/01_causal_chunk_kv_rollout/README.md':'> **版本复核（2026-10-10）：** 本页旧视频实际含fixed-mix adapter，作为工程突破/训练分支证据保留；不能当零训练native V1。新选中的零adapter124f片与最早seed2片已分别收入[V1](../../mainline/V1_native_chunk_causal/README.md)。\n\n',
      'experiments/02_rgb_anchor_visual_drift_repair/README.md':'> **归因边界（2026-10-10）：** 代表性视觉改善来自RGB-consistent anchor、visual QKV与endpoint/boundary等联合协议；下文“根因/修复”是当时局部诊断，不能解释所有崩坏或把全部收益归于anchor。当前[V2版本说明](../../mainline/V2_rgb_anchor_causal/README.md)同时保留20秒失败。\n\n',
      'INTERVIEW_ANSWER.md':'> **2026-10-10更新：** [V0–V4路线](mainline/README.md)与[新汇报视频](report/README.md)已整理。V3是Original权重、Single I0/native时间、Same-σ、T2重算、C12→5自身history两块56f的局部正控；不是V2续训，也不是strict causal/KV或完整AnyFlow/Stage2成功。下文旧“会议主片”始终指V2，E1/E2指对应历史实验。\n\n',
      'REPRODUCE.md':'> **展示材料整理（2026-10-10）：** 现有运行路径、权重和adapter接口保持原样。新[report](report/README.md)可单独复制汇报，其中code是阅读快照，不是独立33B环境；[素材制作与完整性验收](archive/reorganization_20261010/README.md)只使用CPU/PyAV，不运行模型。\n\n',
      'docs/EXPERIMENT_REPORT.md':'> **版本与演示已统一（2026-10-10）：** [mainline](../mainline/README.md)区分V1零adapter、V2联合修复、V3无persistent KV的局部正控；新视频和讲稿在[report](../report/README.md)。下面详细历史结果仍保留原实验scope。\n\n'
    }
    for p,b in banners.items():
        old=(SUB/p).read_text();put(p,b+old);actions.append(dict(operation='add current-navigation/scope banner',source=p,target=p))
    # Gallery navigation is generated from actual successful outputs.
    results=json.loads((AUDIT/'render_results.json').read_text())
    gallery='# 横向比较画廊\n\n全部来自已经保存的真实MP4；左旧右新，A/D分开再合并。V1_vs_V0与V1_vs_original为同义命名副本，其余不是伪造的视频。\n\n| 视频 | 帧数 / 秒 | 用途 |\n|---|---|---|\n'
    for r in results:
        f=r['filename'];m=r['video']
        desc=r.get('title','V0→V1；与Original对照字节相同')
        gallery+=f'| [{f}]({f}) | {m["frames"]} / {m["duration_s"]:.3f} | {desc} |\n'
    gallery+='''
V3主对照只显示Original/V2的RGB[0,56)；两者完整124f源片在各版本videos保留。V3四路径在RGB39高亮chunk边界。A→D/D→A没有匹配Original/V2视频，标为MISSING_MATCHED_VIDEO，不做假配对。

[对比公平性](../COMPARISON_PROTOCOL.md) · [版本变化与未解决问题](../roadmap.md) · [返回汇报](../README.md)
'''
    put('report/00_comparison_gallery/README.md',gallery)
    # Minimal local presentation app, no network or symlinks.
    choices=[('Original→V1','00_comparison_gallery/V1_vs_original.mp4'),
             ('V1→V2','00_comparison_gallery/V2_vs_V1.mp4'),
             ('V3 四路径 / RGB39切块','00_comparison_gallery/V3_four_paths_56.mp4'),
             ('V2→V3 跨协议','00_comparison_gallery/V3_vs_V2.mp4'),
             ('Original→V3 前56帧','00_comparison_gallery/V3_vs_original.mp4'),
             ('20秒完整失败证据','V2_rgb_anchor/videos/V2_long20s_failure.mp4')]
    options=''.join(f'<option value="{path}">{html.escape(name)}</option>' for name,path in choices)
    put('report/index.html','''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3-World × SolarWM | Research Demo</title><style>
body{font:17px/1.65 system-ui,sans-serif;background:#101723;color:#e7edf5;margin:auto;max-width:1180px;padding:24px}
h1{font-size:28px}a{color:#90cfff}select{font:inherit;background:#26374e;color:white;padding:8px;width:100%}
video{display:block;width:100%;max-height:70vh;background:black;margin-top:16px}.note{color:#ffd48c}details{margin:18px 0}
</style><h1>H3-World × SolarWM：因果化与动作信息流</h1>
<p>工程可行；完整动作、长期稳定与加速尚未同时达成。当前最好：<strong>V3 C12→5的第二块局部正控</strong>，没有persistent hidden KV。</p>
<label>选择对比视频 <select id="pick">'''+options+'''</select></label>
<video id="player" controls preload="metadata" playsinline src="00_comparison_gallery/V1_vs_original.mp4"></video>
<p class="note">原速24fps；未补帧或循环。V3跨版只比较前56帧，不是单因素消融。20秒失败片保留完整后段。</p>
<details><summary>建议讲述顺序与版本边界</summary><p>V0动作正控 → V1工程跑通但动作减弱 → V2联合视觉修复、动作未恢复 → V3回到Original恢复原生条件并重算历史，只有第二块局部通过 → V4高效化仍待研究。AnyFlow/DMD是此前独立探索，不是V3已完成Stage1/2。</p></details>
<p><a href="README.md">汇报导航</a> · <a href="roadmap.md">路线</a> · <a href="TALK_5MIN.md">5分钟讲稿</a> · <a href="00_comparison_gallery/README.md">全部画廊</a> · <a href="COMPARISON_PROTOCOL.md">公平性</a></p>
<script>const p=document.getElementById('player');document.getElementById('pick').addEventListener('change',e=>{p.pause();p.src=e.target.value;p.load();});</script></html>
''')
    # All pre-existing top-level evidence collections are explicitly classified.
    catalogue=[]
    for root in ['experiments','reports','meeting']:
        for p in sorted((SUB/root).iterdir()):
            if not p.is_dir() or p.name.startswith('.'):continue
            category='historical evidence, indexed in place'
            if root=='reports' and p.name=='stage1_anyflow':category='Branches A/B/C; retained internal relative runtime paths'
            if root=='meeting':category='legacy presentation; immutable videos/metrics; new report contains curated copies'
            catalogue.append(dict(path=str(p.relative_to(SUB)),policy='KEEP_IN_PLACE',classification=category,
                                   files=sum(q.is_file() for q in p.rglob('*')),
                                   videos=sum(1 for q in p.rglob('*.mp4'))))
    put('archive/legacy_collections.json',json.dumps(catalogue,ensure_ascii=False,indent=2))
    put('archive/README.md','''# Archive · 历史证据与整理审计

- [historical_docs](historical_docs/README.md)：通过git mv归档旧进度、计划和文件清单，旧docs/archive路径保留兼容页。
- [legacy_reports](legacy_reports/README.md)：旧REPORT与原导航，避免“当前”含义混淆。
- [reorganization_20261010](reorganization_20261010/README.md)：整理前文件索引、版本映射、视频来源/hash、复制移动清单、渲染和验收收据。
- [legacy_collections.json](legacy_collections.json)：旧experiments/reports/meeting按证据归档，保留原位置，以免破坏运行依赖和视频/指标相对路径。重复视频不删除，只登记。

“归档”不表示删除，也不表示旧实验被最新结果推翻。原始outputs、checkpoint、CSV/JSON/log和冻结代码仍在原位置；新mainline/branches按研究用途引用它们。

[主线](../mainline/README.md) · [研究分支](../branches/README.md) · [精简汇报](../report/README.md)
''')
    put('archive/reorganization_20261010/layout_actions.json',json.dumps(actions,ensure_ascii=False,indent=2))
    put('archive/reorganization_20261010/README.md','''# 2026-10-10 整理与CPU视频制作审计

先审计Git，再建索引/版本映射，之后复制、git mv历史文档、CPU制作对比。原始测量、checkpoint、视频与生产代码未改写。

- [before_tracked.json](before_tracked.json)：整理前2844个跟踪文件的SHA256；基准commit。
- [file_index.json](file_index.json) / [video_index.tsv](video_index.tsv)：提交包与两处outputs的文件/视频索引（只列物理树，不重复遍历目录软链接）。
- [version_map.json](version_map.json)：复制与渲染之前冻结的版本选择和缺失。
- [selected_sources.json](selected_sources.json)：主展示12条原片、原config/run与SHA。
- [copy_manifest.json](copy_manifest.json) / [layout_actions.json](layout_actions.json)：字节复制与文档移动/跳转。
- [render_specs.json](render_specs.json) / [render_results.json](render_results.json) / [gallery_copies.json](gallery_copies.json)：逐新MP4配方、decode/PTS/hash和独立汇报副本。
- [validation.json](validation.json)：完整性、链接、语法、视频及便携检查；[legacy_link_issues.json](legacy_link_issues.json)单列旧档案原有断链，不改写历史收据掩盖。

CPU制作源码：[prepare_sources](../../scripts/reorganization/prepare_sources.py)、[render_gallery](../../scripts/reorganization/render_gallery.py)、[write_documents](../../scripts/reorganization/write_documents.py)、[finalize_layout](../../scripts/reorganization/finalize_layout.py)、[validate](../../scripts/reorganization/validate.py)。

这些脚本不是模型训练入口。prepare/write/finalize是本次整理的来源记录，finalize不是可重复执行的迁移器；render拒绝覆盖已有MP4。若需重做渲染，先复制仓库到单独目录并使用空gallery，不能覆盖原片。验证可独立重复：

```bash
/home/lpeng/miniconda3/envs/h3world/bin/python submission/scripts/reorganization/validate.py
```

PyAV调用FFmpeg/libx264，Pillow缩放与文字覆盖。只使用CPU；GPU模型参数从未加载。新视频完整解码为H.264/YUV420P/24fps，帧PTS逐项检查；旧文件按SHA比对。
''')
    print(f'Archived {len(moves)} historical documents with git mv; compatibility paths retained.')


if __name__=='__main__':main()
