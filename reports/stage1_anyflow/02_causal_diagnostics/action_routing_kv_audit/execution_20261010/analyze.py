"""CPU-only receipt analysis. No model load or generation."""
from pathlib import Path
import csv,json,statistics
import torch

BASE=Path(__file__).resolve().parent


def geometry(p,q):
    x=(p['A']-p['D']).double().flatten();y=(q['A']-q['D']).double().flatten()
    return dict(cosine=float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-30)),
        norm_ratio=float(x.norm()/y.norm().clamp_min(1e-30)),
        relative_l2=float((x-y).norm()/y.norm().clamp_min(1e-30)))


def main():
    allruns={};p0=[];p1=[];original=[]
    for name in ('run_01','run_02','run_03','run_04'):
        file=BASE/name/'evaluation.json'
        if not file.exists():continue
        r=json.loads(file.read_text());allruns[name]=dict(status=r['status'],model_forwards=r['model_forwards'],
            wall_seconds=r['wall_seconds'],GPU_peak_MiB=r.get('GPU_peak_MiB'))
        for i,row in enumerate(r['P0']):
            si=i%3;ref=row['history']
            pairs={a:torch.load(BASE/name/f'P0_{ref}_{si}_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
            g=geometry({a:pairs[a]['R3_matched'] for a in 'AD'},{a:pairs[a]['R2'] for a in 'AD'})
            p0.append(dict(run=name,history=ref,sigma=row['sigma'],passed=row['passed'],
                max_KV_relative_rms=row['max_KV_relative_rms'],
                max_KV_absolute=max(l[k]['max_abs'] for rec in row['records'] for l in rec['layers'] for k in ('key','value')),
                max_rope_absolute=max(l['rope']['max_abs'] for rec in row['records'] for l in rec['layers']),
                max_velocity_relative_rms=row['max_velocity_relative_rms'],
                max_velocity_absolute=max(rec['velocity']['max_abs'] for rec in row['records']),
                cache_unchanged=row['cache_unchanged'],**{'delta_'+k:v for k,v in g.items()}))
        if name=='run_04':
            for row in r['P1']:
                for k,v in row['vs_Original'].items():
                    original.append(dict(history=row['history'],sigma=row['sigma'],role=k,**v))
                for k,v in row['comparisons'].items():
                    p1.append(dict(history=row['history'],sigma=row['sigma'],comparison=k,**v))
    means={}
    for name in sorted({x['comparison'] for x in p1}):
        rows=[x for x in p1 if x['comparison']==name]
        means[name]=dict(points=len(rows),**{metric:dict(mean=statistics.mean(x[metric] for x in rows),
            min=min(x[metric] for x in rows),max=max(x[metric] for x in rows))
            for metric in ('cosine','norm_ratio','relative_l2')})
    result=dict(status=allruns.get('run_04',{}).get('status'),runs=allruns,P0=p0,P1=p1,P1_summary=means,
        scope='Fixed-state velocity/implementation diagnosis only; not video quality or action-direction acceptance')
    result['Original_reference_summary']={role:dict(points=len(rows),
        **{key:statistics.mean(x[key] for x in rows) for key in ('cosine','norm_ratio','relative_l2')})
        for role in sorted({x['role'] for x in original})
        for rows in [[x for x in original if x['role']==role]]}
    if (BASE/'run_04/evaluation.json').exists():
        result['P2_noised_history']=json.loads((BASE/'run_04/evaluation.json').read_text())['P2']
    result['P2_N_bridge']=[]
    for ref in 'AD':
        bridge=BASE/'N_bridge'/f'{ref}.pt';main=BASE/'run_04'/f'P2_N_{ref}.pt'
        if bridge.exists() and main.exists():
            b=torch.load(bridge,map_location='cpu',weights_only=True)
            m=torch.load(main,map_location='cpu',weights_only=True)
            repeat=float((b['R1',ref]-m['R1'][ref]).abs().max())
            if repeat!=0:raise RuntimeError(('N bridge replay mismatch',ref,repeat))
            p={a:b['R1P',a] for a in 'AD'}
            result['P2_N_bridge'].append(dict(history=ref,sigma=.689441,
                separate_process_R1_replay_max_abs=repeat,
                R1PN_vs_R1N=geometry(p,m['R1']),R2N_vs_R1PN=geometry(m['R2'],p)))
    (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('P0_metrics.csv',p0),('P1_metrics.csv',p1),('Original_reference_metrics.csv',original)]:
        if rows:
            with (BASE/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=result['status'],P0_points=len(p0),P1_comparisons=len(p1),means=means),indent=2))

if __name__=='__main__':main()
