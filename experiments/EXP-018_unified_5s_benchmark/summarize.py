"""Make a lossless compact CSV from per-row metrics; no model calls."""
from __future__ import annotations
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
INDEX=HERE/'results_index.json'
ORDER=('v0','v1','v2a','v2b','v3_30','v3_8','v4')
FIELDS=('row','status','cold_total_seconds','process_to_full_file_seconds_mtime',
    'ready_to_video_seconds','sampling_steps_per_chunk','sampling_nfe_total',
    'commit_forward_total','prefill_forward_total','total_forward_count',
    'peak_allocated_gib','peak_reserved_gib','cpu_kv_peak_gib','load_seconds',
    'conditioning_seconds','sampling_loop_inclusive_seconds','decode_seconds',
    'encode_seconds','video_noise_sha256','full_video','display_video')

index=json.loads(INDEX.read_text())
with (HERE/'metrics_table.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
    writer.writeheader()
    for row in ORDER:
        value=index['rows'].get(row,{'row':row,'status':'NOT_RUN'})
        writer.writerow(value)
print(HERE/'metrics_table.csv')
