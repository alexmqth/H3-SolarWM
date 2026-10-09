"""Verify and archive completed local videos; manual review owns acceptance."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import av

BASE=Path(__file__).resolve().parent
DEST=BASE.parents[3]/'submission/reports/stage1_anyflow/history_conditioning'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=json.loads((BASE/'window1_review.json').read_text())
    assert review['status']=='complete_static_review'
    source_protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,digest in source_protocol['sources'].items():assert sha(BASE/rel)==digest
    for rel,digest in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(BASE/'runtime'/rel)==digest
    gate=json.loads((BASE/'probe_gate.json').read_text());assert gate['diagnostic_forwards']==56
    rows=[];cost=[];audits=[]
    for ref in 'AD':
        group=f'window1_{ref}';folder=BASE/group
        r=json.loads((folder/'evaluation.json').read_text());assert r['status']=='complete_pending_visual_review'
        proc=Path(f'/proc/{r["pid"]}/stat')
        if proc.exists():
            state=proc.read_text().rsplit(')',1)[1].split()
            assert state[19]!=r['start_ticks'] or state[0] in ('Z','X'),'Video process still active'
        assert r['denoiser_forwards']==60 and r['optimizer_updates']==0
        assert r['history_noise_unchanged'] and r['parameter_versions_unchanged']
        assert r['source_sha256']==sha(BASE/'run_video.py')
        assert r['protocol_sha256']==sha(BASE/'protocol.json')
        assert r['video_launch_sha256']==sha(BASE/'video_launch.json')
        assert r['probe_gate_sha256']==sha(BASE/'probe_gate.json')
        old=json.loads((BASE/f'source_coarse/coarse_{ref}/evaluation.json').read_text())
        videos=[]
        for path in sorted(folder.glob('*.mp4')):
            expected=50 if path.name=='CN_AD_context.mp4' else 42
            with av.open(str(path)) as c:
                stream=c.streams.video[0];frames=list(c.decode(video=0))
                assert len(frames)==expected and stream.average_rate==24
                assert stream.codec_context.name=='h264' and all(f.format.name=='yuv420p' for f in frames)
                videos.append(dict(name=path.name,frames=len(frames),width=stream.width,height=stream.height,
                                   codec='h264',pix_fmt='yuv420p',fps=24,sha256=sha(path)))
        assert len(videos)==3
        for n in r['records']:
            a=n['action'];c=next(x for x in old['records'] if x['window']==1 and x['action']==a)
            assert c['history_sha256']==r['history_sha256'] and c['initial_noise_sha256']==r['current_initial_noise_sha256']
            assert c['prompt_sha256']==n['prompt_sha256'] and c['layout_positions_sha256']==n['layout_positions_sha256']
            assert sha(folder/f'{a}.mp4')==n['video_sha256']
            verdict=next(x['observation'] for x in review['branches'] if x['history']==ref and x['action']==a)
            rows.append(dict(history=ref,action=a,C_flow=c['flow']['horizontal_flow_px']['mean'],N_flow=n['flow']['horizontal_flow_px']['mean'],
                             C_frame_MAD=c['frame_gray_MAD'],N_frame_MAD=n['frame_gray_MAD'],
                             C_boundary_MAD=n['boundary_MP4_gray_MAD']['C'],N_boundary_MAD=n['boundary_MP4_gray_MAD']['N'],
                             N_sampling_seconds=n['sampling_seconds'],review=verdict))
        audit=dict(at=datetime.now().astimezone().isoformat(),status='complete_full_decode',group=group,videos=videos,paired_C_N_inputs_verified=True)
        (folder/'video_audit.json').write_text(json.dumps(audit,indent=2)+'\n');audits.append(audit)
        cost.append(dict(history=ref,wall_seconds=r['wall_seconds'],load_seconds=r['load_seconds'],GPU_peak_MiB=r['GPU_peak_MiB'],
                         noisy_forwards=60,CPU_KV_MiB=0,optimizer_updates=0))
        target=DEST/group;target.mkdir(parents=True,exist_ok=True)
        for p in folder.iterdir():
            if p.suffix in ('.json','.jpg','.png','.mp4','.md'):
                shutil.copy2(p,target/p.name);assert sha(p)==sha(target/p.name)
    passed=review['both_histories_action_structure_and_adherence_pass']
    decision='Window1局部门槛通过，窗口2待执行；尚未通过完整E1。' if passed else 'Window1局部门槛失败，停止本历史加噪因素，不执行窗口2，不进入AnyFlow/Stage2。'
    result=dict(at=datetime.now().astimezone().isoformat(),status='reviewed',window1_pass=passed,stage1_accepted=False,
                decision=decision,rows=rows,cost=cost,video_audits=audits,diagnostic_forwards=56,sampling_forwards=120,
                optimizer_updates=0,runtime_unchanged=True,review_method=review['method'])
    (BASE/'video_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table='\n'.join(f'| {r["history"]} | {r["action"]} | {r["C_flow"]:+.5f} | {r["N_flow"]:+.5f} | {r["C_frame_MAD"]:.3f}/{r["N_frame_MAD"]:.3f} | {r["C_boundary_MAD"]:.3f}/{r["N_boundary_MAD"]:.3f} |' for r in rows)
    manual='\n'.join(f'- 历史{r["history"]}／当前{r["action"]}：{r["review"]}' for r in rows)
    timing='\n'.join(f'| {r["history"]} | {r["wall_seconds"]:.2f} | {r["GPU_peak_MiB"]:.2f} | 60 | 0 |' for r in cost)
    text=f'''# E1 历史条件视频结果

{result['at']}。**{decision}** 只更改历史加噪及相应video时间协议；Original权重、native prefix、单I0、12latent、30步、action路由、当前初始noise与保存历史保持。零训练。

## 可播放对比

- [固定A历史，C/N × 当前A/D](window1_A/CN_AD_context.mp4)
- [固定D历史，C/N × 当前A/D](window1_D/CN_AD_context.mp4)

左列C为上一轮clean历史；右列N为同sigma临时加噪历史；上排当前A，下排当前D。开头8帧是两边相同的已知历史末段，随后42帧为当前窗口，便于检查是否有突跳/重置。每个完整网格50RGB，24fps。它不是50帧自由生成，也不是GT条件；reference为Original-generated历史。

## 指标

| 固定历史 | 当前动作 | C水平flow | N水平flow | frame MAD C/N | 首边界MAD C/N |
|---|---|---:|---:|---:|---:|
{table}

Flow沿用原Farneback mean horizontal flow，像素/相邻帧。frame MAD为未压缩当前片段的灰度相邻帧差，属于活动量。首边界MAD使用C/N各自MP4解码第一当前帧与**相同**lossless历史末帧比较；只衡量这个连接，不是画质或全视频连续性。控制来源和视频hash逐项核实。

## 静态逐帧检查

{manual}

详细[人工记录](window1_review.json)。检查全部当前帧、历史末帧及原尺寸细节；不声称实时播放评审。六个MP4已完整解码核验H264/yuv420p/24fps，提交副本sha256与源一致。

## 机制与成本

[同状态探针](PROBE_RESULTS.md)共56次前向，首窗C/N与旧C重放/重复均0，差分统计独立CPU重算一致。新协议使差分改变；该cosine比较失败控制C与N，不是teacher动作正确性。探针的noisy state来自C轨迹，本轮视频才评价N自身轨迹。

| 历史 | 两动作完整作业wall秒 | torch峰值allocated MiB | noisy forwards | CPU hidden KV MiB |
|---|---:|---:|---:|---:|
{timing}

本组120次采样前向＋56诊断，optimizer=0。wall包含两动作、VAE、网格与评测；不是单条视频端到端延迟，也不是warmup后多次均值。T2每sigma重算可见窗口，raw history只读，不宣称persistent-KV加速。305文件runtime、运行源码/协议、参数版本与history/noise hash保持。最多2张项目GPU，本轮已结束；完整Stage1仍未通过。

所有数字：[机器可读总结](video_summary.json)。后续决策按局部动作、人物结构与历史衔接共同确定，不以光流、低MAD或差分范数替代验收。
'''
    (BASE/'VIDEO_RESULTS.md').write_text(text)
    for p in BASE.iterdir():
        if p.is_file() and not p.is_symlink() and p.suffix in ('.py','.json','.md','.log'):
            shutil.copy2(p,DEST/p.name)
    for ref in 'AD':
        target=DEST/f'probe_{ref}';target.mkdir(exist_ok=True)
        shutil.copy2(BASE/f'probe_{ref}/probe.json',target/'probe.json')
    print(json.dumps({'status':result['status'],'window1_pass':passed,'videos':6},ensure_ascii=False))


if __name__=='__main__':main()
