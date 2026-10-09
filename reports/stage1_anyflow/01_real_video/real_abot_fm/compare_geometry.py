"""Compare completed FM checkpoints on identical recorded probe states, on CPU.

Neither cosine nor this input audit decides the video/action acceptance gate.
The recorded anchor protocol is checked by source identity; actual anchor and
teacher output tensors were not hashed, which remains an explicit limitation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return row['clip'], row['chunk'], row['sigma']


def validate(data):
    require(data['status'] == 'complete', 'Probe is not complete')
    require(data['history_source'] in ('gt', 'step00_generated'),
            'Only fixed GT/step00 states can be used for a matched comparison')
    rows = {key(row): row for row in data['records']}
    require(len(rows) == len(data['records']) == 18, 'Requires 18 unique completed cases')
    clips = {k[0] for k in rows}
    sigmas = {k[2] for k in rows}
    require(len(clips) == 2 and len(sigmas) == 3, 'Expected two scenes and three sigmas')
    require(set(rows) == {(clip, chunk, sigma) for clip in clips
                         for chunk in range(3) for sigma in sigmas},
            'Missing scene/chunk/sigma combination')
    audits = {(r['clip'], r['chunk']): r for r in data['cache_audits']}
    require(len(audits) == len(data['cache_audits']) == 6, 'Requires six cache audits')
    require(set(audits) == {(clip, chunk) for clip in clips for chunk in range(3)},
            'Cache audit keys do not match cases')
    for (_, chunk), audit in audits.items():
        require(audit['read_only'] and audit['shared_A_D_cache'] and
                audit['rebuilt_from_raw_history'] and audit['commits'] == 50 * chunk and
                audit['before_sha256'] == audit['after_sha256'], 'KV was not fixed within A/D')
        if chunk == 1:
            require(audit.get('rebuilt_D_history_equals_A') is True,
                    'A/D history reconstruction equality was not verified')
    repeats = [r for r in rows.values() if 'student_repeat_rmse' in r]
    require(len(repeats) == 6, 'Requires six repeated student/teacher controls')
    require(all(r['student_repeat_rmse'] == r['teacher_repeat_rmse'] == 0
                for r in repeats), 'Repeated forward was not identical')
    require(data['model_forwards'] == 84 and data['clean_forwards'] == 8,
            'Forward-count protocol differs')
    return rows


def compare(before, after):
    old, new = validate(before), validate(after)
    require(set(old) == set(new), 'State keys differ')
    for field in ('history_source', 'state_construction', 'teacher_backend', 'source_sha256'):
        require(before[field] == after[field], f'Protocol changed: {field}')
    # History KV must be rebuilt under each checkpoint. It is not expected to
    # have equal content across changed weights; equality is within each pair.
    fixed_fields = ('state_sha256', 'history_sha256', 'pair_sha256',
                    'source_sha256', 'endpoint_source', 'endpoint_sha256')
    rows = []
    teacher_checks = 0
    for state in sorted(old):
        x, y = old[state], new[state]
        for field in fixed_fields:
            require(x[field] == y[field], f'Input changed at {state}: {field}')
        teacher_pairs = [(x['absolute'][a], y['absolute'][a]) for a in 'AD']
        teacher_pairs.append((x['action_delta'], y['action_delta']))
        xf, yf = x['action_delta']['per_latent_frame'], y['action_delta']['per_latent_frame']
        expected_frames = 2 if x['chunk'] == 2 else 5
        require(len(xf) == len(yf) == expected_frames, 'Latent interval length changed')
        teacher_pairs.extend(zip(xf, yf))
        for p, q in teacher_pairs:
            for field in ('teacher_rms', 'teacher_norm'):
                require(math.isclose(p[field], q[field], rel_tol=1e-9, abs_tol=1e-12),
                        f'Frozen teacher scalar changed at {state}: {field}')
                teacher_checks += 1
        row = dict(clip=x['clip'], chunk=x['chunk'], sigma=x['sigma'])
        for tag, record in [('before', x), ('after', y)]:
            delta = record['action_delta']
            row.update({
                f'{tag}_velocity_cosine': mean(record['absolute'][a]['cosine'] for a in 'AD'),
                f'{tag}_delta_cosine': delta['cosine'],
                f'{tag}_delta_norm_ratio': delta['norm_ratio'],
                f'{tag}_delta_relative_error': delta['relative_delta_error'],
                f'{tag}_student_response_relative_to_velocity': delta['student_response_relative_to_velocity'],
                f'{tag}_teacher_response_relative_to_velocity': delta['teacher_response_relative_to_velocity'],
            })
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int))),
                'Nonfinite geometry metric')
        row['delta_cosine_change'] = row['after_delta_cosine'] - row['before_delta_cosine']
        rows.append(row)

    fields = [f'{tag}_{metric}' for tag in ('before', 'after') for metric in
              ('velocity_cosine', 'delta_cosine', 'delta_norm_ratio', 'delta_relative_error',
               'student_response_relative_to_velocity', 'teacher_response_relative_to_velocity')]
    def summarize(items):
        return dict(count=len(items), **{f'mean_{f}': mean(r[f] for r in items) for f in fields},
                    delta_cosine_improved_cases=sum(r['delta_cosine_change'] > 0 for r in items))
    return dict(status='matched_records_verified', before_step=before['step'],
                after_step=after['step'], history_source=before['history_source'],
                summary=summarize(rows), rows=rows,
                by_chunk={str(c): summarize([r for r in rows if r['chunk'] == c]) for c in range(3)},
                by_sigma={str(s): summarize([r for r in rows if r['sigma'] == s])
                          for s in sorted({r['sigma'] for r in rows})},
                input_contract=dict(recorded_state_and_condition_hashes_equal=True,
                    frozen_teacher_scalar_checks=teacher_checks,
                    history_KV_rebuilt_per_checkpoint=True, KV_fixed_within_each_AD_pair=True,
                    anchor_and_teacher_full_tensor_hashes_available=False),
                caveats=[
                    'Anchors are reconstructed by identical source/protocol from the fixed endpoint; actual anchor tensor hashes were not recorded.',
                    'Teacher norms are cross-run controls, not a full teacher output tensor equality proof.',
                    'Endpoint-noise interpolants are not captured solver states.',
                    'Original teacher recomputes bidirectional history; student uses causal clean KV.',
                    'A cosine change does not establish visual quality, pixel action direction or video PASS.',
                    'GT and generated groups change endpoint, history and anchor together; only checkpoint comparisons within a group are matched.',
                ])


def main(args):
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    require(before['step'] == 0 and after['step'] == 48, 'CLI requires actual step00 and step48')
    report = compare(before, after)
    report['receipts'] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                          for name, path in [('before', args.before), ('after', args.after)]}
    report['checkpoint_sha256'] = {name: data['checkpoint_sha256'] for name, data in
                                   [('before', before), ('after', after)]}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['rows'][0]))
        writer.writeheader(); writer.writerows(report['rows'])
    lines = ['# 真实 FM0 → FM48：同状态当前 A/D 动作差分', '',
             f"固定状态来源：**{report['history_source']}**；18个逐一匹配状态。仅统计已完成结果。", '',
             '| 分组 | 点数 | 整体cos：0→48 | A/D delta cos：0→48 | delta相对误差：0→48 |',
             '|---|---:|---:|---:|---:|']
    groups = [('全部', report['summary'])]
    groups.extend((f'chunk {k}', v) for k, v in report['by_chunk'].items())
    groups.extend((f'sigma {float(k):.6f}', v) for k, v in report['by_sigma'].items())
    for name, r in groups:
        cells = [f"{r[f'mean_before_{m}']:.6f} → {r[f'mean_after_{m}']:.6f}"
                 for m in ('velocity_cosine', 'delta_cosine', 'delta_relative_error')]
        lines.append(f"| {name} | {r['count']} | " + ' | '.join(cells) + ' |')
    lines += ['', '每个checkpoint用自己的权重重建历史KV；只要求各自A/D干预期间KV固定，不能要求训练前后的KV逐元素一致。', '',
              '记录的current/history/endpoint/action-pair/source哈希及执行源码一致；冻结teacher整体与逐latent差分的范数标量一致。RGB anchors按同一源码从同endpoint构造，但未保存anchor逐张量hash；teacher也未保存完整输出hash，不夸大为这两者逐bit对照。', '',
              '**本表不自动判定Stage1或视频PASS。** 必须结合GT-history局部视频、generated-history完整视频和停车场A/D方向；whole velocity相似或delta cosine提高均不能单独代替这些验收。', '',
              '[逐状态CSV](metrics.csv) · [完整控制记录及限制](comparison.json)', '']
    (args.out / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
