"""Independently replay saved field statistics before authorizing local videos."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,digest in protocol['sources'].items():assert sha(BASE/rel)==digest,rel
    receipts={};rows=[];cost=[];changed=[]
    for ref in 'AD':
        path=BASE/f'probe_{ref}/probe.json';r=json.loads(path.read_text());assert r['status']=='complete'
        proc=Path(f'/proc/{r["pid"]}/stat')
        if proc.exists():
            state=proc.read_text().rsplit(')',1)[1].split()
            assert state[19]!=r['start_ticks'] or state[0] in ('Z','X'),'Probe still active'
        assert r['protocol_sha256']==sha(BASE/'protocol.json')
        assert r['diagnostic_forwards']==28 and r['optimizer_updates']==0
        assert r['parameter_versions_unchanged'] and r['history_noise_unchanged']
        assert r['preflight']['first_window_identity']['max_abs']==0
        assert len(r['repeat_checks'])==2 and all(x['comparison']['max_abs']==0 for x in r['repeat_checks'])
        fieldpath=BASE/f'probe_{ref}/probe_fields.pt';assert sha(fieldpath)==r['fields_sha256']
        fields=torch.load(fieldpath,map_location='cpu',weights_only=True)
        assert len(fields)==len(r['records'])==6
        measurable=False
        for value,rec in zip(fields,r['records']):
            assert value['window']==rec['window'] and value['step']==rec['step']
            assert value['state_sha256']==rec['current_state_sha256']
            old=torch.load(BASE/f'source_coarse/coarse_{ref}/window{rec["window"]}_solver_states.pt',map_location='cpu',weights_only=True)
            control=next(x for x in old if x['step']==rec['step'])
            replay=(value['CA'].double()-control['velocity_A'].double()).square().mean().sqrt().item()
            assert abs(replay-rec['saved_C_replay']['difference_rms'])<1e-12
            assert replay<=protocol['saved_replay_relative_rms_max']*max(rec['saved_C_replay']['b_rms'],1e-12)
            # Subtract in stored model-output float32 first, matching the recorded estimand.
            dc=(value['CA']-value['CD']).double().flatten()
            dn=(value['NA']-value['ND']).double().flatten()
            diff=(dn-dc).square().mean().sqrt().item()
            ratio=dn.norm().item()/dc.norm().item()
            cosine=torch.dot(dn,dc).item()/(dn.norm().item()*dc.norm().item())
            for actual,key in [(diff,'difference_rms'),(ratio,'norm_ratio_a_over_b'),(cosine,'cosine')]:
                assert abs(actual-rec['delta_N_vs_C'][key])<1e-10,(ref,rec['window'],rec['step'],key)
            measurable |= diff>protocol['measurable_delta_rms_min']
            rows.append(dict(history=ref,window=rec['window'],step=rec['step'],sigma=rec['sigma'],
                             C_delta_rms=dc.square().mean().sqrt().item(),N_delta_rms=dn.square().mean().sqrt().item(),
                             delta_difference_rms=diff,N_over_C_norm=ratio,N_C_cosine=cosine,C_saved_replay_rms=replay))
        assert measurable==r['measurable_change']
        changed.append(measurable);receipts[ref]=sha(path)
        cost.append(dict(history=ref,wall_seconds=r['wall_seconds'],GPU_peak_MiB=r['GPU_peak_MiB'],diagnostic_forwards=28))
    gate=dict(at=datetime.now().astimezone().isoformat(),status='completed_independent_CPU_replay',
              probe_receipts=receipts,diagnostic_forwards=56,optimizer_updates=0,
              source_sha256=sha(Path(__file__)),video_window1_launch_allowed=all(changed),
              action_correctness_proven=False,stage1_accepted=False,rows=rows,cost=cost,
              interpretation='C/N are different history protocols on identical current noisy states. Failed control C is not a trusted teacher. Effect changes justify looking at local videos, not claiming recovered action geometry.')
    (BASE/'probe_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    table='\n'.join(f'| {r["history"]} | {r["window"]} | {r["step"]} / {r["sigma"]:.5f} | {r["C_delta_rms"]:.6f} | {r["N_delta_rms"]:.6f} | {r["N_over_C_norm"]:.4f} | {r["N_C_cosine"]:.4f} | {r["C_saved_replay_rms"]:.2g} |' for r in rows)
    text=f'''# 历史条件同状态探针：执行与干预检查

{gate['at']}。56次真实H3诊断完成，零训练。每一点固定同一reference history、当前A-solver noisy state、past action、I0/layout/audio，仅fork当前A/D。C/N分别代表clean历史与同sigma临时加噪历史。

两路首窗C/N误差0、重复误差0。旧C输出重放通过；独立CPU重读保存field重算差分、范数、cosine，与收据一致。以下cosine只衡量两种协议的差别，**不是对可信teacher的保真度**。

| 历史 | 窗口 | solver索引 / sigma | C动作差分RMS | N动作差分RMS | N/C范数 | N/C差分cos | 旧C重放RMS |
|---|---|---|---:|---:|---:|---:|---:|
{table}

是否允许窗口1局部视频：{gate['video_window1_launch_allowed']}。它只表示执行语义有效、差分变化超过预注册数值门槛，不表示方向恢复。先生成两history×A/D的窗口1，30步共120前向；动作、结构、历史衔接都成立才允许下一窗口。无加宽、无AnyFlow/DMD，最多3GPU。

原始[完整数字与门槛](probe_gate.json)。原始field tensors保留于项目outputs，不放submission。所有当前状态来自旧C实际solver轨迹，不当作N自身轨迹；只有真实N生成才能评价其自身状态分布。
'''
    (BASE/'PROBE_RESULTS.md').write_text(text)
    print(json.dumps({'status':gate['status'],'video_window1_launch_allowed':gate['video_window1_launch_allowed'],'states':len(rows)}))


if __name__=='__main__':main()
