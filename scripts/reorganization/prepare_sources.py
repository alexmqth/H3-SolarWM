"""Index and package existing evidence only. Never imports torch or runs a model."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

SUB = Path(__file__).resolve().parents[2]
ROOT = SUB.parent
AUDIT = SUB / 'archive/reorganization_20261010'
COMMIT = json.loads((AUDIT / 'before_tracked.json').read_text())['commit']
MAIN = ['V0_original_bidirectional', 'V1_native_chunk_causal',
        'V2_rgb_anchor_causal', 'V3_same_sigma_history']
REPORT = ['V0_original', 'V1_native_causal', 'V2_rgb_anchor', 'V3_same_sigma']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    import av
    assets = []
    for v in range(3):
        for a in 'WSAD' if v == 0 else 'AD':
            if v == 0:
                d = ROOT / f'H3-World/outputs/2026-10-01-20/action_{a}_baseline30_124'
                video, run = d/'baseline.mp4', d/'baseline.json'
            elif v == 1:
                d = ROOT / f'H3-World/outputs/2026-10-02-04/action_{a}_base_causal124_retry2'
                video, run = d/'cached.mp4', d/'cached.json'
            else:
                d = ROOT / f'H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/eval124/{a}'
                video, run = d/'cached.mp4', d/'cached.json'
            r = json.loads(run.read_text())
            assert r['status'] == 'complete'
            if v == 1:
                assert r['trained_for_causal'] is False and r['causal_adapter'] is None
            assets.append(dict(id=f'V{v}_{a}', version=v, action=a, source=str(video.relative_to(ROOT)),
                               run=str(run.relative_to(ROOT)), setup=str((d/'setup.json').relative_to(ROOT)),
                               config=json.loads((d/'setup.json').read_text())['config'],
                               metrics=r))
    cb = ROOT/'H3-World/outputs/2026-10-09-22/chunk_partition_cb'
    for h in 'AD':
        for a in 'AD':
            video = cb/f'C_{h}/sigma_noised_{a}_rollout.mp4'
            assets.append(dict(id=f'V3_{h}{a}', version=3, action=f'{h}->{a}',
                               source=str(video.relative_to(ROOT)),
                               run=str((cb/f'C_{h}/evaluation.json').relative_to(ROOT)),
                               setup=str((cb/'protocol.json').relative_to(ROOT)),
                               config=json.loads((cb/'protocol.json').read_text())['fixed']))
    for x in assets:
        p = ROOT/x['source']
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            x['video'] = dict(width=s.width, height=s.height, fps=float(s.average_rate),
                              frames=sum(1 for _ in c.decode(video=0)), codec=s.codec_context.name)
        assert x['video']['frames'] == (56 if x['version'] == 3 else 124)
        assert x['video']['fps'] == 24
        x['sha256'] = sha(p)
    mapping = dict(base_commit=COMMIT, selected_assets=assets,
        decisions={
          'V0': 'Released H3-World; 30 full-horizon steps; source reference, not GT.',
          'V1': 'Zero newly trained adapters; late native latent-dual configuration, NOT earliest seed2 and NOT fixed-mix.',
          'V2': 'RGB-consistent dual + trained visual QKV + fixed-mix action residual + routing changes; joint protocol.',
          'V3': 'Original weights, native Single I0/time, T2 bidirectional visible-window recompute, no persistent hidden KV; only second chunk.',
          'V4': 'Planned; no video/checkpoint.'},
        missing=['V3 matched Original/V2 A->D and D->A full56 videos',
                 'Frozen exact 2026-10-02 V1 source runtime manifest',
                 'V3 third/fourth chunk and full124 results',
                 'V4 implementation/accepted model',
                 'Matched multi-run hardware speed benchmark'],
        comparison_rules=['V0/V1/V2: source124, seed13, same reported first RGB/prompt/constant action; multi-factor comparison.',
                          'V3 versus V0/V2: frames[0,56), full37 reference versus visible12+5; not a single-factor ablation.',
                          'Same seed alone is not proof of identical noise. Retain original protocol/hash receipts.',
                          'Do not fabricate switched-action baselines; show the four V3 paths separately.'])
    # Mapping is written BEFORE copying/reorganization/rendering.
    dump(AUDIT/'version_map.json', mapping)
    copies = []
    def copy(src, dst, kind):
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert sha(src) == sha(dst), f'Refuse overwrite: {dst}'
        else:
            shutil.copy2(src, dst)
        copies.append(dict(source=str(src.relative_to(ROOT)), target=str(dst.relative_to(SUB)),
                           sha256=sha(dst), bytes=dst.stat().st_size, kind=kind))
    for x in assets:
        v = x['version']; rd = SUB/'report'/REPORT[v]; md = SUB/'mainline'/MAIN[v]
        x['packaged_video'] = f'report/{REPORT[v]}/videos/{x["id"]}_full.mp4'
        copy(ROOT/x['source'], SUB/x['packaged_video'], 'byte-identical representative video')
        for key in ['run', 'setup']:
            copy(ROOT/x[key], md/'receipts'/f'{x["id"]}_{key}.json', 'immutable original receipt')
        dump(md/'manifest.json', dict(base_commit=COMMIT, assets=[a for a in assets if a['version']==v]))
    # Frozen code where available; otherwise explicitly a traceable package snapshot.
    files = {
        0: ['code/abot/infer.py','code/abot/abot_action.py','code/diffsynth_h3_action.patch','code/diffsynth_base_commit.txt'],
        1: ['code/causal/h3_cached.py','code/causal/prototype.py','code/causal/benchmark.py','code/diffsynth_causal.patch'],
        2: ['code/causal/h3_cached.py','code/causal/benchmark.py','code/causal/train_online_selfrollout.py'],
        3: ['reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/run.py',
            'reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/interval_forward.py',
            'reports/stage1_anyflow/02_causal_diagnostics/local_topology/runtime/code/causal/local_topology.py',
            'reports/stage1_anyflow/02_causal_diagnostics/local_topology/runtime/code/causal/h3_cached.py']}
    for v, names in files.items():
        provenance = []
        for name in names:
            p=SUB/name
            if not p.exists():
                # Record absent optional file, never invent the snapshot.
                provenance.append(dict(source=name,status='MISSING_SOURCE_SNAPSHOT')); continue
            dst=SUB/'report'/REPORT[v]/'code'/p.name
            copy(p,dst,'core code snapshot; not a standalone runtime')
            provenance.append(dict(source=name,sha256=sha(p),package_commit=COMMIT,
                                   exact_frozen_runtime=v==3,filename=p.name))
        dump(SUB/'report'/REPORT[v]/'code/SOURCE_MANIFEST.json',provenance)
    # Small source receipts are portable; model weights remain canonical.
    for v, srcs in {
      0:['meeting/DEMO_PROVENANCE.json'],
      1:[],
      2:['meeting/DEMO_PROVENANCE.json','reports/visual_drift_repair/VISUAL_DRIFT_REPAIR_REPORT.md'],
      3:['experiments/11_causal_12_then5_selfhistory/protocol.json',
         'experiments/11_causal_12_then5_selfhistory/manifest.json',
         'experiments/11_causal_12_then5_selfhistory/summary.json',
         'reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/source_runtime_manifest.json']}.items():
        for src in srcs:
            copy(SUB/src,SUB/'mainline'/MAIN[v]/'receipts'/Path(src).name,'immutable source receipt')
    copy(ROOT/'H3-World/outputs/2026-10-02-04/action_base_causal124_flow.json',
         SUB/'mainline'/MAIN[1]/'receipts/action_flow.json','existing measurement; unchanged')
    copy(SUB/'meeting/source_metrics/rgb_visual/summary.json',
         SUB/'mainline'/MAIN[2]/'receipts/recorded_metrics.json','existing measurement; unchanged')
    # Earliest causal video stays separate from the chosen latent-dual V1.
    early = ROOT/'H3-World/outputs/2026-09-30-13/benchmark_cached_cpu_124'
    for f in ['cached.mp4','cached.json','setup.json']:
        copy(early/f,SUB/'mainline'/MAIN[1]/'early_seed2'/f,'earliest cached native; separate protocol')
    copy(ROOT/'H3-World/outputs/2026-09-29-20/baseline_50step.mp4',
         SUB/'mainline'/MAIN[0]/'reference50/baseline_50step.mp4','historical 50-step reference; separate protocol')
    # Full long-horizon references and failure evidence: real files for portable report.
    for f in ['original_W_10s_243f.mp4','original_W_20s_481f.mp4']:
        copy(SUB/'meeting/long_horizon'/f,SUB/'report'/REPORT[0]/'videos'/f,'full Original reference')
    copy(SUB/'meeting/long_horizon/original_vs_rgb_visual_W_20s_481f.mp4',
         SUB/'report'/REPORT[2]/'videos/V2_long20s_failure.mp4','uncut negative evidence')
    supplements = {
      'B_fixed_mix_tradeoff_124.mp4':'meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4',
      'B_real_ABot_FM48_39.mp4':'meeting/model_types/02_adaptation_fm0_vs_fm48_39f.mp4',
      'C_anyflow16_4step_failure.mp4':'meeting/model_types/03_anyflow_vs_fm_4step_39f.mp4',
      'C_dmd_lite_failure.mp4':'meeting/stage2_lite/stage2_lite_39_8step_4updates_AD.mp4'}
    for dst, src in supplements.items():
        copy(SUB/src,SUB/'report/01_research_branches_summary/videos'/dst,'representative research-branch evidence')
    dump(AUDIT/'copy_manifest.json',copies)
    dump(AUDIT/'selected_sources.json',assets)
    print(json.dumps(dict(selected_videos=len(assets),copied_files=len(copies),mapping='version_map.json')))


if __name__ == '__main__':
    main()
