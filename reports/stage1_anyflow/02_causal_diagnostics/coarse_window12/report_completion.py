"""Summarize completed E1 coarse controls; never promote a metric to visual PASS."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(path):
    r = json.loads(path.read_text())
    assert r['status'] in ('complete', 'complete_pending_visual_review')
    proc = Path(f'/proc/{r["pid"]}/stat')
    if proc.exists():
        s = proc.read_text().rsplit(')', 1)[1].split()
        assert s[19] != r['start_ticks'] or s[0] in ('Z', 'X'), 'Original process still active'
    return r


def main():
    review = json.loads((BASE/'complete_visual_review.json').read_text())
    assert review['status'] == 'complete_static_review'
    rows, runs = [], []
    for group in ('matched_first5', 'coarse_A', 'coarse_D'):
        path = BASE/group/'evaluation.json'
        r = completed(path)
        for field, filename in [('source_sha256', 'run_windows.py'),
                                ('runtime_manifest_sha256', 'runtime_manifest.json'),
                                ('protocol_sha256', 'protocol.json'),
                                ('launch_protocol_sha256', 'launch_protocol.json'),
                                ('single_anchor_variant_sha256', 'single_anchor.py'),
                                ('prefix_variant_sha256', 'native_prefix.py')]:
            assert r[field] == sha(BASE/filename), (group, filename)
        assert r['parameter_versions_unchanged'] and r['history_latents_unchanged']
        assert r['optimizer_updates'] == 0 and r['cpu_KV_MiB'] == 0
        audit = json.loads((BASE/group/'video_audit.json').read_text())
        assert audit['status'] == 'complete_full_decode'
        for pair in r['local_action_pairs']:
            w = pair['window']
            a, d = [next(x for x in r['records'] if x['window']==w and x['action']==act) for act in 'AD']
            for f in ('history_sha256', 'initial_noise_sha256', 'anchor_sha256', 'layout_positions_sha256'):
                assert a[f] == d[f], (group, w, f)
            if group == 'matched_first5':
                verdict = 'FAIL: A/D nearly identical'
            else:
                verdict = next(x['verdict'] for x in review['windows']
                               if x['group']==group and x['window']==w)
            rows.append(dict(group=group, window=w, RGB_range=a['rgb_range'],
                             A=pair['A'], D=pair['D'], separation=pair['A']-pair['D'],
                             correct_flow_signs=pair['A']>0 and pair['D']<0,
                             frame_gray_MAD_A=a['frame_gray_MAD'], frame_gray_MAD_D=d['frame_gray_MAD'],
                             sampling_seconds_A=a['sampling_seconds'], sampling_seconds_D=d['sampling_seconds'],
                             verdict=verdict))
        runs.append(dict(group=group, wall_seconds=r['wall_seconds'], load_seconds=r['load_seconds'],
                         GPU_peak_MiB=r['GPU_peak_MiB'], CPU_KV_MiB=r['cpu_KV_MiB'],
                         raw_history_MiB=r['latent_history_MiB'], denoiser_forwards=r['denoiser_forwards'],
                         diagnostic_forwards=r['diagnostic_forwards'], clean_commits=r['clean_commits'],
                         receipt_sha256=sha(path), videos=len(audit['videos'])))
    # Preserve the runtime snapshot; mutable outputs are not part of this manifest.
    for rel, expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():
        assert sha(BASE/'runtime'/rel)==expected, rel
    vae = completed(BASE/'gt_boundary_audit_v2/audit.json')
    assert len(vae['records']) == 8
    assert vae['source_sha256']==sha(BASE/'audit_gt_boundaries_v2.py')
    result = dict(at=datetime.now().astimezone().isoformat(),
                  status='complete_no_go', rows=rows, runs=runs,
                  future_RGB_encoder_audit=[{k:q[k] for k in ('clip_id','rgb_stop','latent_stop',
                      'historical_max_abs','historical_mean_abs','safe_vs_full_max_abs','safe_vs_full_mean_abs')}
                      for q in vae['records']],
                  runtime_305_files_unchanged=True, optimizer_updates=0, local_action_gate=False,
                  stage1_accepted=False, same_state_geometry_completed=False,
                  next_decision='Stop window-width expansion; audit history conditioning before another effect-training run. E2 needs credible local consequence pairs; E3 remains gated.')
    (BASE/'final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table = '\n'.join(f'| {r["group"]} | {r["window"]} / [{r["RGB_range"][0]},{r["RGB_range"][1]}) | {r["A"]:+.6f} | {r["D"]:+.6f} | {r["separation"]:.6f} | {r["verdict"]} |' for r in rows)
    cost = '\n'.join(f'| {r["group"]} | {r["denoiser_forwards"]}+{r["diagnostic_forwards"]} | {r["wall_seconds"]:.2f} | {r["load_seconds"]:.2f} | {r["GPU_peak_MiB"]:.2f} | {r["CPU_KV_MiB"]} |' for r in runs)
    audit = '\n'.join(f'| {q["clip_id"]} | {q["rgb_stop"]}/{q["latent_stop"]} | {q["historical_max_abs"]:.6g} | {q["safe_vs_full_max_abs"]:.6g} |' for q in vae['records'])
    text = f'''# E1 粗窗口完整结果：首窗改善，跨历史局部动作与结构仍 No-Go

{result['at']}。Original H3 + released action LoRA，native text/action time，单 I0，30 steps，零训练。

## 决策

12-latent 当前窗口恢复了首窗口的方向，接入已知历史后仍未同时满足动作与人物结构门槛。停止继续增加窗口宽度，不用 AnyFlow 或 Stage2 处理尚未成立的局部正控。没有证据否定所有 causal 拓扑；这是本组零样本、有限场景下的负结果。

## 动作与画面

| 组 | 窗口 / RGB区间 | A flow | D flow | A−D | 逐帧评审 |
|---|---|---:|---:|---:|---|
{table}

flow 为既有 Farneback mean horizontal flow，单位 px/相邻帧。不是因果准确率；A>0、D<0只对本停车场正控使用，必须结合画面。MAD是活动量，不是画质。

两份reference的首窗都没有历史，不是两个独立泛化样本。[同前17RGB结果](FIRST_WINDOW_RESULTS.md)控制了统计长度，但12latent仍可使用整个当前已知窗口生成/解码，不代表相同控制延迟。所有分支共用history、初始noise、I0与布局哈希；A/D去噪轨迹随后分叉。本次没有把各自solver轨迹相减作同状态velocity证据，也没有完成新的same-state geometry。

人工评审：[完整记录](complete_visual_review.json)。检查完整静态帧和关键帧原尺寸细节，不冒称实时播放评审。MP4均完整解码核验。[固定A历史，当前A/D](coarse_A/AD_local_forks.mp4)；[固定D历史，当前A/D](coarse_D/AD_local_forks.mp4)。

这些120f视频是39+42+39帧的oracle局部串接，每窗恢复Original-generated参考历史；不是GT，也不是124f自由rollout。边界重置不作长时连续性证据；没有替换会议最终demo。

## 成本与执行核验

| 组 | noisy + diagnostic forwards | 全作业wall s | 模型加载 s | torch峰值allocated MiB | CPU hidden KV MiB |
|---|---:|---:|---:|---:|---:|
{cost}

全作业wall包含同一进程内A/D多个分支、VAE、编解码/静态图与评测，不是一条视频的端到端推理耗时。最多3张项目GPU；共享主机、没有warmup后重复，不提供speedup。逐分支采样时间、frame MAD和raw-history大小见[完整数据](final_summary.json)。权重/激活峰值尚未单独分解。

T2每sigma重算可见history/prefix，CPU hidden KV=0，不能冒称persistent-KV加速；raw history latent另记。参数版本、历史hash、305文件runtime及启动源码/协议hash均保持。全部模型任务已退出，420 noisy+9 diagnostic forwards，optimizer=0。

## GT历史编码的未来RGB干预

| 验证片段 | RGB前缀/latent前缀 | 反转未来RGB后past latent最大差 | 仅过去RGB补末帧 vs 全片prefix最大差 |
|---|---:|---:|---:|
{audit}

只加载VAE、无DiT/optimizer/下载。第一尝试D_1750原片剩余长度不足，加载模型前退出；[失败原记录](gt_boundary_audit/audit.json)保留。第二尝试换同验证episode的A_1030，纯RGB因果审计，不当作A/D配对。完整[审计](gt_boundary_audit_v2/audit.json)。干预结果仅对本布局/实现/两样本负责，不把尚未观察到的泄漏当失败根因；当前模型实验使用已有generated latent history，与GT编码审计不同。

## 下一步

区分已证实事实与待检验解释：首窗恢复、后窗失败已观察到；clean-history噪声/时间条件、位置、历史动作影响仍有混杂，尚未定位唯一根因。[CPU实际历史条件审计](history_contract_audit.json)的9状态确认历史video time=1，而全部text/action time=1−sigma；它符合现有pipeline/retake语义，不直接判bug。下一项[历史条件有限对照](NEXT_HISTORY_CONTROL.md)已设计、尚未实现/启动GPU；仍用Original权重、30步、固定12latent窗口/单I0，不扩大网络或窗口。

E2需要可靠的局部动作后果参照，不能用本轮失真的分支强行监督。E3仍为可信causal → AnyFlow → on-policy Stage2，不要求先解决20秒自由漂移，但首个局部验收尚未通过。
'''
    (BASE/'FINAL_RESULTS.md').write_text(text)
    print(json.dumps({'status':result['status'],'rows':len(rows),'runs':len(runs),'vae_records':len(vae['records'])}))


if __name__ == '__main__':
    main()
