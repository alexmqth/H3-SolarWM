"""Build the research navigation and portable presentation from audited evidence."""
from pathlib import Path
import json
import os
import re
import subprocess

SUB=Path(__file__).resolve().parents[2]
AUDIT=SUB/'archive/reorganization_20261010'
MAIN=['V0_original_bidirectional','V1_native_chunk_causal','V2_rgb_anchor_causal','V3_same_sigma_history']
REP=['V0_original','V1_native_causal','V2_rgb_anchor','V3_same_sigma']
BASE=json.loads((AUDIT/'before_tracked.json').read_text())['commit']
ASSETS=json.loads((AUDIT/'selected_sources.json').read_text())


def put(rel,text):
    p=SUB/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.rstrip()+'\n')


def link(src,target,label=None):
    return f'[{label or Path(target).name}]({os.path.relpath(SUB/target,(SUB/src).parent)})'


TABLE='''| 版本 | Action fidelity | Visual stability | 长时 rollout | Persistent KV | Sampling steps | 新增训练 | 验收状态 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D 正控；W/S 以视频为准 | 124f 基本完整 | 有243/481f参考；仍有几何变形 | 否 | 30整段；另存历史50步 | 无，released LoRA | Reference，不是GT |
| V1 Native causal | A/D显著减弱；本代表A符号错 | 停滞、透明/重影等退化 | 124f执行完成 | 是，CPU raw video KV | 8/chunk × 8 + 8 commits | 本代表无 | 工程可行，质量/动作未过 |
| V2 RGB联合修复 | A符号错；未恢复 | 124f相对改善；20s失败 | 124/243/481f均有片 | 是，CPU raw video KV | 8/chunk | visual QKV + action residual | 视觉局部改善，联合验收失败 |
| V3 Same-σ C12→5 | 两份自身history下当前A/D方向正确 | 第二块人物结构保持 | 仅56f、2块 | **否**，T2逐sigma联合重算 | 30/chunk | 无，从Original重新出发 | **Current Best Local Causal Rollout Candidate** |
| V4 Efficient causal | 待验证 | 待验证 | 待验证 | 目标：是 | 先30/chunk可信，再少步 | 待定 | Planned，无checkpoint/视频 |
'''

SECTIONS=[
 dict(name='V0 · Original H3-World (Bidirectional Baseline)',
 objective='保留原始 H3-World 的动作和画面正控，作为后续协议的 reference。',
 parent='公开 MiniMax-H3 底座 + released H3-World action LoRA；不需要重训 bidirectional pretrained model。',
 changes='完整视频联合去噪；原生 Single-Egress：Vᵢ直接读Aᵢ，Aᵢ只直接读对应video；跨时间动作可通过video hidden传播。',
 config='主片：832×480，124 RGB / 24fps，seed13，30 full-horizon steps，flow shift2.22，Single I0。50-step旧片另存为历史reference，不能与主片合并统计。',
 quant='124f A/D horizontal flow：+1.0767 / −1.6019，separation 2.6786。39f旧teacher的2.023是另一长度/协议，不拼成同一学习曲线。',
 improved='提供动作方向及人物结构正控，也保留了10s/20s原始生成参考，撤回“Original天然不能生成长片”的旧说法。',
 failed='非GT；原始长片也有场景几何变形。此处没有因果交互、历史KV复用或首屏低延迟证明。',
 lesson='先固定正控，再区分attention、时间、图像条件和采样预算的影响；不能仅因seed相同就宣称噪声完全相同。',
 source='H3-World/outputs/2026-10-01-20/action_{A,D,W,S}_baseline30_124/；released LoRA SHA256 ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526。',
 videos=[('V0_AD_reference.mp4','A/D原始方向正控'),('V0_WSAD_reference.mp4','W/S/A/D原始总览'),('original_W_10s_243f.mp4','Original 10秒完整参考'),('original_W_20s_481f.mp4','Original 20秒完整参考')]),
 dict(name='V1 · Native Chunk-Causal H3-World',
 objective='验证 Original 权重能否按chunk生成并使用真正的persistent raw KV，展示直接因果化带来的动作/画面问题。',
 parent='V0权重。主代表明确选零新增adapter的native后期latent-dual协议，不是fixed-mix，也不是最早seed2版本。',
 changes='5 latent/chunk，generated history，CPU raw video KV，clean commit和5-chunk窗口；dynamic_last_frame_dual。own action prefix / feedback off；固定prefix时间，与原生时间不同。',
 config='2026-10-02 action_{A,D}_base_causal124_retry2；124f/24fps，seed13，8steps/chunk，shift2.22，64 noisy forwards + 8 clean commits；trained_for_causal=false，causal_adapter=null。',
 quant='原记录A=−0.017959、D=−0.025755，A−D=0.007796；A/D E2E分别465.9/400.9s，CPU KV13.19GiB，GPU allocated peak约39.01GiB。这些为共享硬件单次记录。',
 improved='逐块执行、clean commit、历史复用和完整124帧输出已实现。更早seed2/4-step版本另存，不以新配置改写早期证据。',
 failed='本代表A/D运动几乎消失、A符号错；抽帧显示人物停滞、后段过亮和背景退化。更严重的人物撕裂/重影见另存的fixed-mix及早期DMD片，不能归成同一个checkpoint。',
 lesson='Causalization是协议改造，不自动保留Original的动作传播。persistent KV执行成功也不等于动作或质量成功。',
 source='H3-World/outputs/2026-10-02-04/action_{A,D}_base_causal124_retry2/。无新增训练checkpoint；历史原始底座与released LoRA仍为外部依赖。',
 videos=[('V1_vs_original.mp4','Original vs V1，A/D完整124f'),('V1_vs_V0.mp4','V0→V1；与前一文件字节相同的命名别名'),('V1_A_full.mp4','V1 A原片'),('V1_D_full.mp4','V1 D原片')]),
 dict(name='V2 · RGB-Anchor Causal H3-World',
 objective='修正latent-only条件与RGB视觉训练条件不一致的问题，改善V1家族的generated-history视觉退化。',
 parent='研究上承接V1问题；权重来自RGB tail16 visual adaptation + fixed-mix action residual，不是本次选中的V1零adapter直接续训。',
 changes='生成prefix latent→VAE decode→取末RGB→process_image=True重编码；RGB-consistent dual anchor。联合tail16 QKV普通teacher replay、Original endpoint和boundary监督；causal action prefix + feedback。',
 config='visual_online_rgb_tail16_endpoint_ad2；尾16 QKV两次online更新，冻结fixed-mix action residual；124f/24fps，seed13，8/chunk×8，CPU KV。没有AnyFlow或DMD student更新。',
 quant='124f A=−0.7841、D=−1.0075，separation0.2233；A/D E2E699.5/715.9s，CPU KV13.19GiB。新主表RGB MAD3.1855/3.0356、boundary3.7871/3.8874；与旧报告的另一MAD实现分开，未改写原值。',
 improved='所选124f中人物和停车场结构相对视觉崩坏版本保持得更好。改善属于RGB协议、visual adapter和监督等联合方案，不能全部归功于anchor。',
 failed='A/D仍同为负，动作验收失败；10s后段退化，20s约10秒起雾化，15秒后人物/场景难辨。保留完整失败视频。',
 lesson='视觉稳定与动作保真是不同验收轴。对齐image-conditioning可修复一部分结构问题，不能替代动作信息流恢复。',
 source='checkpoints/visual_rgb_tail16/{causal_adapter.pt,action_adapter.pt}；原实验H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/。adapter未复制进汇报包。',
 videos=[('V2_vs_original.mp4','Original vs V2，A/D124f'),('V2_vs_V1.mp4','V1→V2，联合协议对比'),('V2_long20s_failure.mp4','同checkpoint20秒完整失败证据'),('V2_A_full.mp4','V2 A原片'),('V2_D_full.mp4','V2 D原片')]),
 dict(name='V3 · Same-σ History / C12→5 H3-World',
 objective='Current Best Local Causal Rollout Candidate：验证首12latent正控接自身history后，第二个5latent的动作与结构。',
 parent='**从Original H3 + released action LoRA重新出发**，没有使用V2 visual adapter，也不是V2 checkpoint的后续训练。',
 changes='Single I0、native action/time、固定全局RoPE；首12后续5；历史按当前sigma与固定历史噪声临时加噪；T2每步联合重算可见history/current。未来action/video在refiner前物理移除。',
 config='30steps/chunk，shift2.22，seed13；39帧自身生成首段+17帧续写=56f/24fps。T2窗口内video双向；只积分当前chunk，不更新保存的过去。没有persistent hidden KV或GT reset。',
 quant='第二块：A-history下当前A/D为+2.1464/−1.7519；D-history下+2.6076/−1.4143。每条续写30 forwards，sampling约194.6–195.5s，GPU peak25.42GiB，CPU hidden KV=0。首窗重用，不能将续写时间当全56f E2E。',
 improved='两份自身history下第二块A/D方向正确，人物结构保持，无瞬间换场；相较clean history在D→A的方向失败，这是明确的局部正结果。',
 failed='只有停车场、seed13、第二块；没有第三/第四块、跨场景或完整124f验收。不是strict chunk-causal mask，更没有高效persistent KV证明。',
 lesson='V3是历史协议/可见窗口的新候选，不是Same-σ单因素修好了V2。decoder重解码可能修改过去末5RGB；展示冻结已发39帧再追加17帧，边界不能隐藏。',
 source='H3-World/outputs/2026-10-09-22/chunk_partition_cb/；候选C12_then5_N_30step_selfhistory是推理配置，不是LoRA checkpoint。保存的first12/next5 .pt是生成latent endpoints。',
 videos=[('V3_four_paths_56.mp4','四路径完整56f；RGB39边界高亮'),('V3_vs_original.mp4','Original vs V3，AA/DD共同前56f'),('V3_vs_V2.mp4','V2→V3，跨协议研究对比'),('V3_AA_full.mp4','A→A原片'),('V3_AD_full.mp4','A→D原片'),('V3_DA_full.mp4','D→A原片'),('V3_DD_full.mp4','D→D原片')])]


def version_docs():
    fields=[('1. Version Name / Research Objective','objective'),('2. Parent Version / Baseline','parent'),
            ('3. Main Changes','changes'),('4. Model and Inference Configuration','config'),
            ('6. Quantitative Results','quant'),('7. What Was Improved','improved'),
            ('8. What Still Failed','failed'),('9. Lessons Learned','lesson'),
            ('10. Source Code / Checkpoint / Original Experiment References','source')]
    for v,s in enumerate(SECTIONS):
        out=f'report/{REP[v]}/README.md'
        text=f'# {s["name"]}\n\n'
        for i,(heading,key) in enumerate(fields):
            if i==4:
                text+='## 5. Representative Videos\n\n'+''.join(f'- [{label}](videos/{fn})\n' for fn,label in s['videos'])+'\n'
            text+=f'## {heading}\n\n{s[key]}\n\n'
        text+='[核心代码说明](code/README.md) · [代码SHA与源路径](code/SOURCE_MANIFEST.json) · [本版来源清单](PROVENANCE.json) · [返回汇报导航](../README.md)\n\n'
        text+='指标均为历史记录；flow是运动proxy，MAD是活动量/连续性描述，不是视频质量评分。Original生成视频不是GT。\n'
        if v==3:
            text+='\n**MISSING_MATCHED_VIDEO**：Original/V2的A→D和D→A匹配56f视频未找到；四路径只作V3内部展示。AA/DD跨版本主片只用RGB[0,56)，V0/V2完整124f原片仍保留。\n'
        put(out,text)
        codes=json.loads((SUB/f'report/{REP[v]}/code/SOURCE_MANIFEST.json').read_text())
        code_text='# 核心实现阅读快照\n\n这些文件供汇报与代码定位，不是可直接启动33B的独立runtime。完整运行仍依赖底座、released LoRA、DiffSynth和原始运行配置。\n\n'
        for c in codes:
            if 'filename' in c:code_text+=f'- [{c["filename"]}]({c["filename"]})：来源 `{c["source"]}`。\n'
        code_text+='\n'+('V3文件取自已冻结的实验runner/runtime，哈希可对原launch/runtime manifest核验。源码中的外部路径保持原样，不在汇报包里启动。'
                    if v==3 else '这是提交包commit '+BASE+' 的可追溯实现快照。V0/V1早期逐次运行未全部保存冻结runtime，不能把现行文件冒充当时逐字节源码；V2主要接口也以原配置和checkpoint哈希为准。')
        code_text+='\n\n[SOURCE_MANIFEST](SOURCE_MANIFEST.json)记录逐文件SHA256。[返回版本说明](../README.md)。\n'
        put(f'report/{REP[v]}/code/README.md',code_text)
        provenance=dict(version=f'V{v}',package_commit=BASE,
            experiment_sources=[a for a in ASSETS if a['version']==v],
            code_snapshot=codes,checkpoints=s['source'],portable_report_contains_model_weights=False)
        put(f'report/{REP[v]}/PROVENANCE.json',json.dumps(provenance,ensure_ascii=False,indent=2))
        main=f'mainline/{MAIN[v]}/README.md'
        # Same 10-section template, with links rebased from portable report to mainline.
        txt=re.sub(r'\]\(([^)]+)\)',lambda m:']('+os.path.relpath((SUB/out).parent/m.group(1),(SUB/main).parent)+')',text)
        txt+='\n## 完整证据与边界\n\n'+link(main,f'mainline/{MAIN[v]}/manifest.json','原实验/配置/视频对应manifest')+' · '+link(main,'archive/reorganization_20261010/version_map.json','整理前冻结的版本映射')+'\n\n'
        if v==0:txt+=link(main,'mainline/V0_original_bidirectional/reference50/baseline_50step.mp4','另存历史50-step原片')+'；与30步主片不混算。\n'
        if v==1:
            txt+=link(main,'mainline/V1_native_chunk_causal/early_seed2/cached.mp4','最早seed2/4-step cached视频')+'，其'+link(main,'mainline/V1_native_chunk_causal/early_seed2/setup.json','原配置')+'独立保留。早期scheduler修正前后的结果不可混成同协议。\n\n'
            txt+=link(main,'branches/B_causal_adaptation/02_history_mixing/README.md','fixed-mix训练分支')+'的A/D0.4532属于新增adapter；不是本V1主片的0.0078。\n'
        if v==2:txt+=link(main,'reports/visual_drift_repair/VISUAL_DRIFT_REPAIR_REPORT.md','原视觉修复报告')+' · '+link(main,'meeting/DEMO_PROVENANCE.md','checkpoint来源')+' · '+link(main,'meeting/long_horizon/README.md','长片完整负结果')+'\n'
        if v==3:txt+=link(main,'reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/README.md','C/B实验完整报告')+' · '+link(main,'experiments/11_causal_12_then5_selfhistory/manifest.json','生成endpoint状态manifest')+' · '+link(main,'branches/A_causal_diagnostics/04_kv_and_topology/README.md','严格KV对照为什么另立')+'\n'
        put(main,txt)
    put('mainline/V4_efficient_causal_planned/README.md','''# V4 · Efficient Causal H3-World — Planned

1. **Version Name / Research Objective**：将V3可信局部动作与视觉能力迁移到高效因果骨干。
2. **Parent Version / Baseline**：V3推理协议与Original权重正控；不是已完成的新checkpoint。
3. **Main Changes**：计划研究strict chunk-causal attention、明确的public-prefix规则、原生action路由和历史时间、persistent KV。
4. **Model and Inference Configuration**：待定；先30-step可信causal，再AnyFlow，最后on-policy DMD。
5. **Representative Videos**：**NOT_YET_VALIDATED / 无视频**，不从V3复制冒充V4。
6. **Quantitative Results**：无模型性能结果。
7. **What Was Improved**：已有机制审计给出了需要控制的变量，不等于V4能力通过。
8. **What Still Failed**：strict causal + persistent KV尚无动作/画面联合PASS。
9. **Lessons Learned**：先证明相同计算图下cache/recompute等价，再评判局部动作；cosine不等于质量。
10. **Source / Checkpoint / References**：[下一步验收](../../report/02_next_steps/README.md)；**无checkpoint**。

本次仅整理材料；不自动启动任何下一步实验。
''')


BRANCHES={
 'A_causal_diagnostics':[
 ('01_chunk_partition','Chunk 5/7/12 与原生时间网格','implementation verified; V3 local PASS only',
  ['reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/README.md','reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/FINAL_RESULTS.md','reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/README.md'],
  'B7首窗A方向失败；C12→5 + N在第二块局部通过。均不能推广到124f。'),
 ('02_history_anchor','Clean/Same-σ 与 Single I0 / latent / RGB','mixed; protocol-dependent',
  ['experiments/09_original_conditioning_calibration/README.md','experiments/10_history_conditioning_action_response/README.md','reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/VIDEO_RESULTS.md'],
  '恢复原生条件取得首窗正控；Same-σ对历史续写有效，但更早12→12仍有ghosting。RGB联合修复不能当anchor单因素结论。'),
 ('03_action_routing','Single-Egress / feedback / public prefix','implementation verified; action mismatch diagnosed',
  ['reports/stage1_anyflow/02_causal_diagnostics/first_chunk_routes/README.md','reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md','reports/action_alignment/ACTION_ROUTING_PROBE.md'],
  'causal action prefix允许当前V直接读past A；raw cache仅存video K/V，额外action直读来自fresh prefix。own仍允许通过历史video间接传播。'),
 ('04_kv_and_topology','Recompute / Persistent KV / Strict video causalization','implementation verified under controlled numerics',
  ['reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md','reports/stage1_anyflow/02_causal_diagnostics/local_topology/README.md'],
  '统一SDPA可见组与GEMM形状后，6状态×A/D的50层K/V、RoPE和velocity误差0。无persistent KV的R2对Original动作差分cos已0.057；生产默认不声称bitwise等价。'),
 ('05_vae_time_rope','VAE边界 / native time / global RoPE / future leakage','implementation verified; visual attribution limited',
  ['reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/vae_audit.json','reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/README.md','reports/stage1_anyflow/04_numerical_checks/README.md'],
  '12latent prefix前34RGB与完整decode一致，末5RGB可能回改。V3冻结已输出39帧再追加；action-video索引未发现切错，不以此排除所有表示问题。')],
 'B_causal_adaptation':[
 ('01_tail_qkv','Tail-QKV ordinary causal adaptation','preliminary trained; joint quality failed',
  ['docs/CAUSAL_BASELINE.md','meeting/model_types/README.md','code/causal/train_pretrained_multichunk.py'],
  '普通FM causal adaptation已训练，并不等于AnyFlow；加载训练容器也不代表optimizer已经更新。'),
 ('02_history_mixing','Fixed-mix / scheduled sampling','preliminary trained; visual/action trade-off',
  ['checkpoints/legacy_fixed_mix/README.md','meeting/action_vs_stability/README.md','meeting/diagnostics/legacy_fixed_mix/README.md'],
  '旧fixed-mix124f A/D=+0.1427/−0.3105，比V2符号好但尾段撕裂；不是Original级控制，也不是零训练native V1。'),
 ('03_online_replay','Online per-sigma teacher replay','preliminary trained; quality failed',
  ['reports/action_alignment/ACTION_ALIGNMENT_REPORT.md','reports/action_alignment/ACTION_ALIGNMENT_LEARNING_CURVE.md','code/causal/train_online_selfrollout.py'],
  '逐sigma replay及paired训练链路存在；内部loss/cosine改善没有可靠转化为自由rollout方向与视觉联合PASS。'),
 ('04_action_geometry','Action velocity / score-delta / endpoint','diagnosed; recovery failed',
  ['reports/teacher_delta_audit/COMBINED_REPORT.md','reports/action_alignment/FINAL_ACTION_DIAGNOSTIC.md','reports/endpoint_target/ACTION_QKV_OWN_ENDPOINT_REPORT.md','experiments/05_corrected_own_history_endpoint/README.md'],
  '区分同一history/noisy state的反事实差分与不同轨迹相减。多种action路径适配未过，不能用差分cosine独自认定视频好。'),
 ('05_rgb_visual','RGB visual + Original endpoint supervision','preliminary trained; 124f visual improved, action failed',
  ['reports/visual_drift_repair/VISUAL_DRIFT_REPAIR_REPORT.md','meeting/DEMO_PROVENANCE.md'],
  '对应V2联合协议；普通replay/endpoint两更新，不是Stage2。Stage2-lite后来集成是另一个实验。'),
 ('06_real_abot_fm','Real ABot FM48 / noise-density FM48','preliminary trained; quality failed',
  ['reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md','reports/stage1_anyflow/01_real_video/fm_density_control/FINAL_RESULTS.md','meeting/DATASET_AND_ANYFLOW.md'],
  '真实录制ABot：16训练/8验证，按episode拆分，48更新；objective为ordinary FM。GT history与generated history分别验收，后者仍分解。'),
 ('07_e2_action_ranking','FM-only vs FM+Action consequence ranking','preliminary trained; no consistent gain; frozen',
  ['reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md','reports/stage1_anyflow/01_real_video/real_transition_windows/FREEZE.json','reports/stage1_anyflow/01_real_video/real_transition_windows/SUPERVISION_COVERAGE.md'],
  'A-history、D-history和两条真实GT-history全收齐；两臂各4更新，无一致额外收益。没有自动扩到16。')],
 'C_anyflow_dmd':[
 ('01_tf_anyflow','TF-AnyFlow实现与数值验证','implementation verified; full Stage1 not accepted',
  ['code/causal/ANYFLOW_PROVENANCE.md','reports/stage1_anyflow/04_numerical_checks/README.md','code/causal/anyflow.py','code/causal/anyflow_sampling.py'],
  '有限区间finite-map、目标时间条件和r=t/符号/梯度等检查存在；ordinary FM与AnyFlow目标明确区分。'),
 ('02_anyflow16','AnyFlow16 vs matched FM16','preliminary trained; quality failed',
  ['reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/FINAL_RESULTS.md','reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/FINAL_RESULTS.md','meeting/model_types/README.md'],
  '两条Original A/D伪标签；16更新；4/8steps/chunk。训练初始化来自旧RGB causal协议，不是V3。等更新不等算力。'),
 ('03_anyflow64_128_136','AnyFlow64/128/136 budgets','preliminary trained; quality failed',
  ['reports/stage1_anyflow/03_anyflow_trials/full_history_duration64/FINAL_RESULTS.md','reports/stage1_anyflow/05_runtime/parallel_resume68_to128/STEP128_RESULTS.md','reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md'],
  '这些预算实际存在。128的8步A−D0.7597，136两臂0.5810/0.6352；4步后段雾化，内部一致性下降不构成成功。'),
 ('04_finite_numerics','Finite-interval / diagonal / teacher endpoint','implementation verified; ability not established',
  ['reports/stage1_anyflow/03_anyflow_trials/dual_metric136/README.md','reports/stage1_anyflow/03_anyflow_trials/finite_interval_probe128/README.md','reports/stage1_anyflow/03_anyflow_trials/diagonal_inference128/FINAL_RESULTS.md'],
  '同时保留finite-vs-diagonal自一致性和对Original teacher endpoint的误差；两者都不单独等于视觉质量。'),
 ('05_real_data_objectives','Real ABot与AnyFlow/FM数据划分','objective audited; real ABot AnyFlow run not found',
  ['meeting/DATASET_AND_ANYFLOW.md','reports/stage1_anyflow/01_real_video/README.md','reports/stage1_anyflow/03_anyflow_trials/VIDEO_INDEX.md'],
  '目录名stage1_anyflow不代表全是AnyFlow。已查训练：AnyFlow用Original伪标签，真实ABot主要FM48/E2；未找到真实ABot AnyFlow训练结果，不能编造。'),
 ('06_stage2_lite','Trainable fake score / self-rollout / DMD surrogate','implementation verified; preliminary trained; quality failed',
  ['experiments/03_stage2_lite_critic_dmd/README.md','code/causal/stage2_lite_dmd.py','meeting/stage2_lite/'],
  '共享H3的student/teacher/critic及实际小预算DMD链路存在，旧latent版人物分解，RGB集成A/D仍未恢复。不称完整SolarWM Stage2成功。'),
 ('07_dmd_preparation','Shared roles / FMBS / DMD gradients','implementation verified on small tests; 33B quality not validated',
  ['reports/stage1_anyflow/06_stage2_preparation/README.md','reports/stage1_anyflow/06_stage2_preparation/dmd_gradient/README.md','reports/stage1_anyflow/06_stage2_preparation/h3_fmbs_integration/README.md'],
  'CPU/小H3梯度、角色隔离和FMBS工程测试不升级为完整33B Stage2。V3还未接AnyFlow或DMD。')]
}


def branch_docs():
    index=[]
    for branch,items in BRANCHES.items():
        text=f'# {branch}\n\n按研究问题归档；下表区分实现、训练完成和能力验收。原报告/原始日志保留原路径与字节。\n\n| 子实验 | 状态 | 结论 |\n|---|---|---|\n'
        for slug,title,status,refs,conclusion in items:
            rel=f'branches/{branch}/{slug}/README.md'
            text+=f'| [{title}]({slug}/README.md) | {status} | {conclusion} |\n'
            detail=f'# {title}\n\n**状态：{status}**\n\n{conclusion}\n\n## 原报告、数据、代码与视频\n\n'
            for r in refs:
                assert (SUB/r).exists(),r
                detail+='- '+link(rel,r,r)+'\n'
            detail+='\n视频及逐运行指标沿用上述原报告内的真实链接。完整文件/视频索引：'+link(rel,'archive/reorganization_20261010/video_index.tsv','整理前视频索引')+'。\n\n'
            detail+='[返回本分支](../README.md)；本次没有新训练或模型推理。\n'
            put(rel,detail)
            entry=dict(branch=branch,experiment=slug,title=title,status=status,conclusion=conclusion,evidence=refs)
            put(f'branches/{branch}/{slug}/manifest.json',json.dumps(entry,ensure_ascii=False,indent=2));index.append(entry)
        text+='\n[返回研究分支总览](../README.md) · [主线版本](../../mainline/README.md) · [精简汇报](../../report/README.md)\n'
        put(f'branches/{branch}/README.md',text)
    put('branches/README.md','''# Research Branches

这些是主线的机制诊断与先行训练探索，不是一条已经全部成功的checkpoint继承链。

- [A · Causal Mechanism Diagnostics](A_causal_diagnostics/README.md)：分块、条件、路由、KV、VAE与时间位置。
- [B · Causal Adaptation & Action Recovery](B_causal_adaptation/README.md)：FM适配、mix/replay、action监督、RGB视觉修复和真实ABot。
- [C · AnyFlow & DMD Explorations](C_anyflow_dmd/README.md)：finite-map数值、16/64/128/136试验及critic/DMD准备。

状态：`implementation verified`只说明接口/数值；`preliminary trained`只说明做过更新；`quality failed`表示未过对应动作/视觉门槛；`not yet validated`表示缺少能力证据。可同时出现，不能用“训练完成”替代“能力通过”。

[逐子实验机器索引](EXPERIMENT_INDEX.json) · [完整旧档案入口](../reports/stage1_anyflow/README.md) · [主线](../mainline/README.md) · [汇报包](../report/README.md)
''')
    put('branches/EXPERIMENT_INDEX.json',json.dumps(index,ensure_ascii=False,indent=2))


def landing_docs():
    put('report/README.md','''# H3-World × SolarWM：一分钟汇报导航

**目标：** 将SolarWM的因果分块、KV cache与少步思路迁入H3-World，改善长视频效率，同时保留action control。

**当前结论：** strict causal + persistent KV已在真实33B上跑通；RGB联合协议在124帧改善视觉，却没有恢复A/D控制，20秒仍失败。最新V3从Original恢复原生条件、使用Same-σ和C12→5联合重算，在**第二块**同时取得局部动作与人物结构正结果。还没有“动作对、长期稳、整体更快”的完整模型。

**建议播放顺序：**

1. [Original vs V1：直接因果化丢失了什么](00_comparison_gallery/V1_vs_original.mp4)。
2. [V1→V2：视觉联合修复与动作局限](00_comparison_gallery/V2_vs_V1.mp4)。
3. [V3四路径：自身history后的局部正结果](00_comparison_gallery/V3_four_paths_56.mp4)，黄色边框从RGB39开始。
4. [V2→V3：前56帧跨协议对比](00_comparison_gallery/V3_vs_V2.mp4)。V3没有继承V2的visual adapter。
5. [保留的20秒失败证据](V2_rgb_anchor/videos/V2_long20s_failure.mp4)。后段完整，没有裁掉崩坏。

'''+TABLE+'''
## 按需展开

- [V0 Original](V0_original/README.md) / [V1 Native causal](V1_native_causal/README.md) / [V2 RGB联合修复](V2_rgb_anchor/README.md) / [V3 Same-σ](V3_same_sigma/README.md)：统一十项说明、核心源码与原片。
- [完整横向比较画廊](00_comparison_gallery/README.md)：包括逐A、逐D和合并片；无新模型采样。
- [研究路线](roadmap.md) / [三个研究分支与失败实验](01_research_branches_summary/README.md)。
- [下一步与Go/No-Go](02_next_steps/README.md) / [5分钟答辩稿](TALK_5MIN.md)。
- [单次历史性能与口径](METRICS.md) / [对比公平性与缺失](COMPARISON_PROTOCOL.md)。

**当前最好的是V3局部候选，不是最终高效模型。** 下一步缺的是将这项局部能力迁回strict causal/KV，再验证多块，随后才有可靠的AnyFlow与on-policy DMD基础。本次只整理现有结果，没有启动这些研究。

本目录可独立复制用于汇报：视频均为实际文件，源码是阅读快照，不含33B权重。大规模原始证据留在完整仓库；各版本PROVENANCE.json记录源位置/配置/hash。所有时间为原共享主机单次记录，没有宣称warmup多次均值或公平speedup。
''')
    put('mainline/README.md','# Mainline · 模型与协议演进\n\n版本号描述研究路线，不保证checkpoint逐代继承。尤其V3从Original重新出发。\n\n'+TABLE+'\n'+''.join(f'- [{SECTIONS[i]["name"]}]({d}/README.md)\n' for i,d in enumerate(MAIN))+'- [V4 Efficient causal — Planned](V4_efficient_causal_planned/README.md)\n\n[精简汇报](../report/README.md) · [研究分支](../branches/README.md)\n')
    put('report/roadmap.md','''# 研究路线与版本关系

V0 Original H3-World → V1 Native Causalization → V2 Visual Conditioning Repair → V3 Same-σ Action Recovery（仅局部） → V4 Efficient Causalization（Future） → AnyFlow Few-step Distillation → On-policy DMD。

上面是问题演进顺序，**不是同一checkpoint连续训练链**。

```mermaid
flowchart TD
    V0["V0 Original H3 + released action LoRA"] --> V1["V1 Strict chunk-causal + persistent KV"]
    V1 --> V2["V2 RGB conditioning + visual adaptation"]
    V0 --> V3["V3 Single I0 / native time / Same-sigma / C12→5"]
    V2 -. "视觉与动作的取舍促成重新审视协议；不继承权重" .-> V3
    V3 --> V4["V4 Planned: preserve capability under strict causal + KV"]
    V4 --> AF["Future: AnyFlow few-step"]
    AF --> DMD["Future: on-policy DMD"]
    A["Branch A: causal mechanism diagnostics"] -.-> V4
    B["Branch B: causal adaptation and action recovery"] -.-> V2
    C["Branch C: preliminary AnyFlow / DMD explorations"] -. "已有探索，非V3完成Stage1/2" .-> AF
```

| 迭代 | 上一阶段的主要失败 | 新阶段改变了什么 | 已解决与未解决 |
|---|---|---|---|
| V0→V1 | 完整horizon联合去噪，不能直接按块复用历史 | causal分块、raw KV、clean commit、generated history、后期latent dual | 工程rollout可执行；动作及局部结构丢失 |
| V1→V2 | V1家族人物透明、重影、分裂，动态条件与视觉训练不一致 | RGB decode/re-encode + visual QKV + endpoint/boundary + routing联合协议 | 124f视觉改善；动作失败，20s崩坏；不能归因anchor单项 |
| V2→V3 | 视觉/动作取舍，strict causal依赖图与原动作传播不匹配 | 回到Original，无visual adapter；Single I0/native time；T2每sigma重算，Same-σ，C12→5 | 自身history第二块动作及人物结构局部通过；无persistent KV/长期验证；跨协议比较 |
| V3→V4 | 只有慢速局部候选；严格图/KV未保留该能力 | **计划**隔离public prefix、action routing、历史时间、strict图和cache | 尚无V4结果 |

## 三个旁路与正确的后续顺序

Branch A厘清Action–Video配对、公共prefix反馈、video图和KV冻结；Branch B测试ordinary FM、mix、teacher replay、动作监督和真实ABot，得到正负结果；Branch C提前探索AnyFlow和DMD，实现/数值可行但质量未过。它们均没有证明V3已经完成AnyFlow/Stage2。

AnyFlow学习有限区间flow map，目的为减少每chunk采样次数；DMD让student在自己生成的分布上得到teacher与trainable fake score之差的训练方向。DMD不能被当作任何causal topology或action表示错误的万能修复。

[下一步验收](02_next_steps/README.md) · [回到首页](README.md)
''')
    put('report/02_next_steps/README.md','''# V4及之后：研究决策与验收

本页是roadmap，不是已经启动的任务。没有新训练、GPU推理、AnyFlow或DMD进程。

| 顺序 | 研究问题 | Go标准 | 当前缺口 |
|---|---|---|---|
| 1 | V3局部正结果能否接到更多自身history | 同协议第三/第四块、至少额外场景，A/D方向与人物结构共同评审 | 尚无这些视频 |
| 2 | V4能否保留Original action信息流 | 相同权重/状态/时间/可见输入；own路由和feedback/public-prefix策略明确，局部30-step通过 | 无strict causal+KV联合PASS |
| 3 | KV实现是否与同计算图重算一致 | 跨sigma逐层K/V、RoPE、velocity；区分matching-time rebuild与sigma0 clean commit | 诊断wrapper已证明受控等价；生产默认数值/重算接口仍需谨慎 |
| 4 | AnyFlow能否少步且保能力 | 4/8步接近已可信causal30；finite/diagonal与teacher endpoint双指标，完整视频复核 | 旧AnyFlow质量失败；V3未进行此阶段 |
| 5 | on-policy DMD能否减少generated-history gap | clean-history局部能力可信；真实self-rollout、frozen teacher、trainable fake score、DMD梯度 | 已有lite负结果和小模型工程准备，非完整成功 |
| 6 | 是否更快/更省 | 统一硬件/offload、warmup、多次均值；E2E、首屏实际可见延迟、每块、GPU/CPU KV | 当前只有共享主机单次耗时，不授予加速结论 |

过去审计的关键事实：cache只存video K/V；causal action prefix额外直读来自fresh prefix。统一数值路径下P0逐层误差为0，但R2无persistent KV时对Original动作差分cosine已约0.057。因此不能将动作失败全部归于缓存实现；也不能仅提高cosine就宣布画面恢复。

[研究路线](../roadmap.md) · [汇报导航](../README.md)
''')


def extra_report_docs():
    put('report/COMPARISON_PROTOCOL.md','''# 视频比较范围与公平性

所有新视频只从已生成MP4取真实RGB帧：24fps、不补帧、不慢放、不循环；等比例缩放留边、横向左旧右新、统一0-based RGB帧号。H.264/YUV420P/faststart，逐帧解码验证。静音是原生成协议；没有合成画面。

| 比较 | 显示帧段 | 相同项 | 不同项 / 限制 |
|---|---|---|---|
| Original vs V1 | [0,124) | 停车场、首图/prompt记录、常量A/D、seed13、832×480源 | 30整段 vs 8/chunk，Single I0 vs latent dual，prefix时间/feedback/attention，不能解释为仅加KV |
| Original vs V2 | [0,124) | 同场景/首图/prompt/action/seed/源分辨率；旧收据有噪声一致性说明 | visual+action adapters、RGB dual、routing/时间/采样协议；Original非GT |
| V1 vs V2 | [0,124) | chunk5、history5、8/chunk、CPU KV、常量动作、seed13 | 无新增adapter vs visual+action训练，latent vs RGB，own/off vs causal/on；联合方案比较 |
| Original/V2 vs V3 | [0,56) | 常量A或D、同停车场首图/seed13、源832×480、24fps | full124生成后裁56 vs12+5可见窗口；精度、时间、anchor、routing、history、步数不同；V3不继承V2权重 |
| V3四路径 | [0,56)；39处边界 | 每行同自身generated first12，当前A/D反事实，共享保存噪声 | 只有第二块；跨两行history本来不同，不直接相减作为同状态velocity |

Same seed不自动证明noise逐字节一致，尤其不同张量长度。V3来自固定full37 fixture，源protocol保存输入与runtime hash；V0/V2旧比较有自己的来源核验。本次只核验文件与收据，没有加载大tensor伪造新的张量匹配证明。

**MISSING_MATCHED_VIDEO**：没有找到与V3 A→D / D→A相同39帧切换点的Original和V2完整56f视频。不会使用恒定A/D冒充这些切换基线。V3与Original/V2仅比较AA/DD；其余两条单独展示。

V3的首39帧是已保存的自身生成首段，随后追加RGB39:56；不是GT reset，也不是把Original full-horizon输出拷作history。VAE重新decode可能改写过去末5帧，本展示按原实验冻结已发帧，保留边界跳变。

V3跨版展示是研究方案比较，不能归因Same-σ单项；V3不是strict chunk-causal attention或persistent KV的成功证据。所有完整源片同时保留，裁剪对比不会隐藏V2的124/481f负结果。

视频中的时间来自原运行收据：V0/V1/V2标完整124f E2E，即使展示只裁56f；V3标第二块sampling并注明重用首窗。均不是制作MP4所花时间。

[画廊](00_comparison_gallery/README.md) · [指标口径](METRICS.md) · [首页](README.md)
''')
    text='# 主要历史指标（不新增模型测量）\n\n统一摘录原收据，保留作用范围。下面是共享主机单次记录，非统一offload/warmup多次均值，不能直接排名speedup。\n\n| 版本/动作 | 范围 | 时间s / scope | GPU peak GiB | CPU KV GiB | noisy+commit | flow x |\n|---|---|---:|---:|---:|---|---:|\n'
    original=json.loads((SUB/'meeting/source_metrics/rgb_visual/summary.json').read_text())
    for v in [0,1,2]:
        for a in 'AD':
            x=next(x for x in ASSETS if x['id']==f'V{v}_{a}');r=x['metrics']
            flow=next(z['horizontal_flow_mean'] for z in original['actions'] if z['action']==a and z['method']==('original' if v==0 else 'causal')) if v!=1 else {'A':-.01795905276,'D':-.02575477814}[a]
            text+=f'| V{v} {a} | 124f全片 | {r["wall_including_shared_setup_seconds"]:.1f} E2E | {r["memory_entire_run_peak_MiB"]/1024:.2f} | {r.get("kv_cache_peak_MiB",0)/1024:.2f} | {r["denoiser_forwards"]}+{r.get("commit_forwards",0)} | {flow:+.4f} |\n'
    summ=json.loads((SUB/'experiments/11_causal_12_then5_selfhistory/summary.json').read_text())
    for r in summ['metrics']:
        if r['mode']!='sigma_noised':continue
        text+=f'| V3 {r["case"][-1]}→{r["action"]} | 第二块17f | {r["sample_seconds"]:.1f} sampling | {r["GPU_peak_MiB"]/1024:.2f} | 0 | 30+0 | {r["flow_mean_x"]:+.4f} |\n'
    text+='\nV3复用首39帧，不把第二块sampling冒充全片E2E；有些诊断forward时间另记。CPU KV=0不表示CPU模型权重/状态为0。GPU为torch allocated峰值，不是整卡占用，也未分解为权重/KV/激活。\n\n'
    text+='8steps/chunk×8chunks=64 noisy forwards，再加8clean commits；Original30整段forwards每次序列更长。步数30/8不是端到端加速比。首块latent计时也不等于用户能看到首屏：原benchmark末尾统一decode。\n\n'
    text+='flow是Farneback中央裁剪水平位移proxy，A正/D负须与人物和场景一起看；不能评判W/S前后移动。V3是17新RGB段，不能与124全段flow比较恢复百分比。MAD低可能只是画面模糊/静止，不是更高质量。未计算FVD/LPIPS/PSNR/VBench，Original也不是GT。\n\n'
    text+='原始收据：V0/V2 `meeting/source_metrics/rgb_visual/summary.json`；V1 `action_base_causal124_flow.json`与每运行cached.json；V3 `experiments/11_causal_12_then5_selfhistory/summary.json`。各版PROVENANCE保存选中运行完整指标/配置。\n\n[返回首页](README.md)\n'
    put('report/METRICS.md',text)
    put('report/01_research_branches_summary/README.md','''# 三个研究分支：已做什么，失败说明什么

| 分支 | 完成的工作 | 状态与结论 |
|---|---|---|
| A Causal Mechanism Diagnostics | 5/7/12分块，clean/N，I0/anchor，action/public prefix，KV/time/RoPE/VAE | 实现与固定状态诊断通过；Original动作传播在strict图中改变，非单纯cache错误。默认数值路径和受控零误差路径明确分开 |
| B Causal Adaptation & Action Recovery | tail QKV、fixed-mix、online replay、action-delta、RGB+endpoint、真实ABot FM48/E2 | 确实训练过；有视觉改善或局部动作变化，但没有一致动作+视觉+长时通过 |
| C AnyFlow & DMD Explorations | TF-AnyFlow有限区间与r=t，16/64/128/136训练、4/8步；fake-score critic/DMD-lite；FMBS/梯度准备 | implementation verified / preliminary trained；旧方案quality failed；完整V3 AnyFlow/Stage2 not yet validated |

## 可现场播放的证据

- [fixed-mix动作较强、视觉较差；RGB反向取舍](videos/B_fixed_mix_tradeoff_124.mp4)：Original / fixed-mix / RGB三列，124f，不是V1零训练主片。
- [真实ABot FM48](videos/B_real_ABot_FM48_39.mp4)：Original / FM0 / FM48，停车场评测；FM48训练数据是真实ABot，而所展示场景是停车场外部正控。
- [AnyFlow16 vs matched FM16，4步失败](videos/C_anyflow16_4step_failure.mp4)：39f完整后段保留。伪标签来自两条Original A/D，不是ABot真实AnyFlow训练。
- [旧Stage2-lite critic/DMD失败](videos/C_dmd_lite_failure.mp4)：4 updates的小预算探索，人物分解；后续RGB集成改善结构但没有恢复动作。不能称完整SolarWM Stage2。

64/128/136均有实际结果和训练记录；增加预算没有把4步质量与A/D联合门槛做过。真实ABot目录虽放在stage1_anyflow历史根下，已核对主要objective是FM或FM+action，不凭目录名升级为AnyFlow。

完整证据在仓库`branches/`及历史`reports/`，汇报包仅保留上面的代表片。[首页](../README.md) · [路线](../roadmap.md)
''')
    put('report/TALK_5MIN.md','''# 5分钟答辩：以技术判断为主

**0:00–0:45：目标与最直接的证据。** 播放[Original vs V1](00_comparison_gallery/V1_vs_original.mp4)。我们把因果分块和KV接入33B H3，但这里A/D运动几乎消失；工程跑通和动作保真是两项结论。

**0:45–1:40：因果与KV怎么实现。** 当前chunk只读取允许的历史video K/V，按层保存raw pre-RoPE K/V及位置；去噪结束clean commit，历史放CPU并按窗口淘汰。Original单出口action路由与causal-prefix不是同一信息流：后者增加past-action直接访问。缓存只存video，额外动作直读来自fresh prefix。

**1:40–2:20：为什么少步不等于更快。** 124帧8chunks×8steps=64 noisy forwards，另有8commits；Original是30次长序列forward。序列长度、重算、CPU offload、RGB反复encode/decode都影响E2E。没有统一硬件重复均值，不能按30/8声称加速。

**2:20–3:10：失败推动了什么改进。** 播放[V1→V2](00_comparison_gallery/V2_vs_V1.mp4)，解释RGB条件、visual QKV与endpoint的联合修复；A方向仍错。再展示[20秒失败片](V2_rgb_anchor/videos/V2_long20s_failure.mp4)后半段，明确不是长视频稳定成功。fixed-mix动作较强却视觉漂移；AnyFlow和DMD-lite也没有解决整体问题。

**3:10–4:20：当前最有价值的局部结果。** 播放[V3四路径](00_comparison_gallery/V3_four_paths_56.mp4)，在RGB39处暂停说明：前39帧是自己生成，第二块只换当前A/D，无GT reset。Original权重、Single I0/native时间、Same-σ、T2联合重算，得到第二块动作方向和人物结构正结果。但没有persistent KV，只有两块，不宣称124帧成功。

**4:20–5:00：研究判断。** 受控数值下cache/recompute逐层完全等价，但无persistent KV的strict图已与Original动作差分失配，说明多层传播/公共prefix/历史时间值得逐项研究。下一步应保留V3局部能力做V4高效因果化；其后AnyFlow减少采样，再on-policy DMD处理生成历史分布。当前成果是明确可行性、失败边界与受控正线索，不是完整复现SolarWM。

可追问：本次所有对比是旧MP4的CPU拼接；不同协议明确标注，不把velocity cosine当视频质量，也不把smoke test当正式成功。

[导航](README.md)
''')


def main():
    version_docs();branch_docs();landing_docs();extra_report_docs()
    print('Version, branch, roadmap and presentation documents written.')


if __name__=='__main__':main()
