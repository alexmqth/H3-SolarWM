"""Summarize complete receipts without promoting diagnostic footage/models."""
import csv
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parent
receipts={a:json.loads((OUT/f'probe_{a}.json').read_text()) for a in ('A','D')}
assert all(d['status']=='complete' and len(d['records'])==6 for d in receipts.values())
rows=[];paths=[]
for history,d in receipts.items():
    assert d['parameter_versions_unchanged']
    assert len(d['cache_audits'])==2 and all(r['unchanged'] for r in d['cache_audits'])
    for r in d['records']:
        matched=r['instantaneous']['matched'];native=r['instantaneous']['native']
        row=dict(history=history,chunk=r['chunk'],sigma=r['sigma'],
            matched_cosine=matched['cosine'],native_cosine=native['cosine'],
            student_delta_rms=matched['student_rms'],matched_teacher_delta_rms=matched['teacher_rms'],
            matched_norm_ratio=matched['norm_ratio'],native_norm_ratio=native['norm_ratio'],
            student_relative_action_effect=matched['student_response_relative_to_velocity'],
            teacher_relative_action_effect=matched['teacher_response_relative_to_velocity'],
            finite_vs_diagonal_cosine=r['finite_vs_diagonal_delta']['cosine'],
            finite_vs_diagonal_norm_ratio=r['finite_vs_diagonal_delta']['norm_ratio'],
            student_repeat_rmse=r.get('student_repeat_rmse'),teacher_repeat_rmse=r.get('teacher_repeat_rmse'),
            future_only_rmse=r.get('future_only_rmse'))
        rows.append(row)
        if 'pathway' in r:
            for name,metric in r['pathway'].items():
                paths.append(dict(history=history,chunk=r['chunk'],pathway=name,**metric))
with (OUT/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
analysis=dict(scope='AnyFlow128, two saved generated histories, 12 explicitly noised states; no optimizer',
    rows=rows,pathways=paths,
    summary={key:dict(mean=mean(r[key] for r in rows),min=min(r[key] for r in rows),
                      max=max(r[key] for r in rows)) for key in
        ('matched_cosine','native_cosine','matched_norm_ratio','native_norm_ratio')},
    controls={key:[r[key] for r in rows if r[key] is not None] for key in
        ('student_repeat_rmse','teacher_repeat_rmse','future_only_rmse')},
    forwards={a:dict(noisy=d['noisy_forwards'],clean=d['clean_commit_forwards'],
        peak_GPU_MiB=d['gpu_allocated_peak_MiB'],peak_CPU_KV_MiB=d['cpu_KV_peak_MiB'],
        wall_seconds=d['wall_seconds']) for a,d in receipts.items()})
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
lines=['# Same-generated-state current-chunk action geometry: measured results','',
    'No training or video acceptance. All 12 local cases completed at AnyFlow128.',
    'These are explicitly noised generated endpoint interpolants, not saved solver intermediates.',
    'The main comparison is instantaneous student `r=t` versus Original H3 instantaneous velocity.',
    '', '| History | Chunk | sigma | Cosine, matched teacher | Cosine, native teacher | Student/teacher norm, matched | Student delta RMS | Teacher delta RMS |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['history']} | {r['chunk']} | {r['sigma']:.6f} | {r['matched_cosine']:.6f} | {r['native_cosine']:.6f} | {r['matched_norm_ratio']:.6f} | {r['student_delta_rms']:.6f} | {r['matched_teacher_delta_rms']:.6f} |")
lines += ['', '## Controls and provenance', '',
    '- Current spans only: latent rows 5–9 / 10–11. Head/past/future rows remain fixed.',
    '- Actual SDPA masks checked at layers 0/25/49: current/past action visibility, own-frame feedback, historical video reads.',
    '- Full KV hashes and commit counts unchanged across the counterfactual branches; parameter versions unchanged.',
    '- Released H3 action LoRA retained; our visual/Stage1 bank/AnyFlow/residual adaptations disabled only for teacher forwards.',
    '- Teacher uses the same generated prefix/noisy chunk, without future video. Matched profile uses identical dual anchors/global positions/fixed prefix times; native profile uses single initial anchor/native times.',
    '- Architectural differences remain: recomputed bidirectional teacher history versus frozen student KV; teacher own-action direct binding versus student past/current action visibility.',
    '- A/D spans have identical saved lengths. The future-action control tests embedding-content leakage at FIXED layout, not arbitrary changes in future sentence lengths.',
    '- Latent rows 5–9 map to RGB [17,34); 10–11 to RGB [34,39). Global current-video positions and the tail-anchor position are preserved (see temporal_layout.json). This does not imply strict frame causality inside a chunk or hard RGB boundaries through the temporal VAE.',
    '', '```json',json.dumps(analysis['controls'],indent=2),'```','',
    '## Middle-sigma pathway intervention','',
    'Each entry isolates one input while holding the other fixed. Cosine refers to the matched teacher delta.',
    '', '| History | Chunk | Isolated path | Effect RMS | Effect/teacher norm | Cosine |',
    '|---|---:|---|---:|---:|---:|']
for r in paths:
    lines.append(f"| {r['history']} | {r['chunk']} | {r['pathway']} | {r['student_rms']:.6f} | {r['norm_ratio']:.6f} | {r['cosine']:.6f} |")
lines += ['', '## Execution and limits', '', '```json',json.dumps(analysis['forwards'],indent=2),'```','',
    'One GPU at a time, CPU raw KV and CPU weight offload. These diagnostic timings are not an inference benchmark.',
    'Finite interval deltas and per-latent-frame metrics are preserved in the receipts. Finite delta versus instantaneous teacher is not a like-for-like velocity comparison.',
    'One seed and scene. A nonzero local action response does not prove correct image-space strafe direction, video stability or that Stage2 will fix the issue. No new demo was generated or promoted.','']
(OUT/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(analysis['summary'],indent=2))
