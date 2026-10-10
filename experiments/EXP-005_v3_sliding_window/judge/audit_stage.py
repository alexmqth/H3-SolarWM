"""Independent stage ledger/cache/condition audit of completed GPU outputs."""
from pathlib import Path
import json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('stage',choices=['G1','L1']);args=p.parse_args()
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parents[2]
OUT=ROOT/'H3-World/outputs/EXP-005_v3_sliding_window'/args.stage
result=json.loads((OUT/'result.json').read_text());budget=json.loads((OUT/'budget.json').read_text())
assert result['status']==budget['status']=='complete_pending_judge_review'
expected=123 if args.stage=='G1' else 124
assert budget['total_forwards']==expected and budget['sampling']==120 and budget['vae']==4
assert budget['commit']==(3 if args.stage=='G1' else 2)
assert budget.get('diagnostic',0)==(0 if args.stage=='G1' else 2)
assert budget['elapsed_seconds']<=2520
assert result['shared_cache_unchanged'] and result['frozen_parameters_unchanged']
assert result['peak_allocated_gib']<=44
checks=[];rows={(r['action'],r['index']):r for r in result['chunks']};assert len(rows)==4
for (action,index),row in sorted(rows.items()):
 assert row['sampling_forwards']==30 and row['vae_calls']==1 and row['cache_read_unchanged'] and row['old_RGB_unchanged']
 assert row['commit_forwards']==(1 if index==6 else 0)
 c=row['cache_before'];assert c['ancestors']==list(range(index-5,index))
 assert c['layer_count']==50 and c['rows_per_layer']==9750 and c['nbytes']==14164800000
 if index==6:
  after=row['cache_after_commit'];assert after['ancestors']==[2,3,4,5,6] and after['nbytes']==c['nbytes']
 start,stop=row['interval'];endpoint=OUT/action/f'chunk_{start}_{stop}.pt'
 assert hashlib.sha256(endpoint.read_bytes()).hexdigest()==row['endpoint_sha256']
 audit=json.loads((HERE/f'{args.stage}_{action}_C{index+1}.json').read_text())
 assert audit['prefix_unchanged'] and audit['RGB_sha256']==row['published_RGB_sha256']
 assert audit['video_frames']==(141 if index==6 else 158)
 checks.append({'action':action,'chunk':index+1,'ancestors':c['ancestors'],'KV_bytes':c['nbytes'],
   'sampling_seconds':row['sampling_seconds'],'commit_seconds':row.get('commit_seconds',0),
   'decode_seconds':row['vae_seconds'],'chunk_wall_seconds':row['chunk_wall_seconds'],
   'flow_horizontal_mean':row['flow']['horizontal_flow_px']['mean'],
   'boundary_gray_MAD':row['boundary_gray_MAD'],'inside_gray_MAD':row['inside_gray_MAD'],
   'peak_allocated_gib':row['peak_allocated_gib'],'peak_reserved_gib':row['peak_reserved_gib'],
   'cpu_rss_mib_after':row['cpu_rss_mib_after']})
for index in (6,7):
 for field in ['noise_sha256','position_sha256']:
  assert rows['A',index][field]==rows['D',index][field]
assert rows['A',6]['history_sha256']==rows['D',6]['history_sha256']
matched_G_L=None
if args.stage=='L1':
 g=json.loads((OUT.parent/'G1/result.json').read_text());gr={(r['action'],r['index']):r for r in g['chunks']}
 for key in rows:
  for field in ['noise_sha256','prompt_sha256','position_sha256']:
   assert rows[key][field]==gr[key][field],(key,field)
  if key[1]==6:assert rows[key]['history_sha256']==gr[key]['history_sha256']
  assert rows[key]['local_oldest_latent']==(12 if key[1]==6 else 17)
 matched_G_L='noise/action/canonical-layout identical; same C7 clean history and shared Global cache; C8 histories branch'
 d=result['fixed_G1_history_C8_position_diagnostic'];assert d['history_unchanged']
out={'stage':args.stage,'independent_audit_passed':True,'checks':checks,'total_forwards':expected,
 'sampling':120,'commit':budget['commit'],'diagnostic':budget.get('diagnostic',0),'vae':4,
 'gpu_seconds':budget['elapsed_seconds'],'gpu_hours':budget['elapsed_seconds']/3600,
 'peak_allocated_gib':result['peak_allocated_gib'],'cpu_peak_rss_mib':result['cpu_peak_rss_mib'],
 'matched_G_L':matched_G_L,'capability_requires_visual_judge':True}
(HERE/f'{args.stage}_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
