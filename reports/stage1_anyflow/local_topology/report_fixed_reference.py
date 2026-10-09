"""Archive completed E1 fixed-text-time controls without claiming acceptance."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import av

BASE=Path(__file__).resolve().parent
DEST=BASE.parents[3]/'submission/reports/stage1_anyflow/local_topology'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[];audits=[]
    for name in ('positive_A','positive_D','window12_calibration'):
        folder=BASE/name;receipt=json.loads((folder/'evaluation.json').read_text())
        assert receipt['status']=='complete_pending_visual_review' and receipt['optimizer_updates']==0
        assert receipt['parameter_versions_unchanged']
        if name.startswith('positive'):
            assert receipt['denoiser_forwards']==180 and len(receipt['records'])==6
            assert receipt['history_latents_unchanged']
            for i in range(3):
                a,d=[next(r for r in receipt['records'] if r['chunk']==i and r['action']==act) for act in 'AD']
                assert a['initial_noise_sha256']==d['initial_noise_sha256']
                assert a['history_sha256']==d['history_sha256']
                assert a['anchor_sha256']==d['anchor_sha256']
                fa,fd=[r['flow']['horizontal_flow_px']['mean'] for r in (a,d)]
                rows.append(dict(reference=receipt['history_reference'],chunk=i,A=fa,D=fd,separation=fa-fd,
                    same_history_noise_anchor=True))
        videos=[]
        for path in sorted(folder.glob('*.mp4')):
            with av.open(str(path)) as container:
                stream=container.streams.video[0];frames=list(container.decode(video=0))
                assert stream.codec_context.name=='h264' and stream.average_rate==24
                assert all(f.format.name=='yuv420p' for f in frames)
                if path.name.startswith('chunk'):
                    index=int(path.name[5]);expected=(17,17,5)[index]
                elif '_' in path.stem and path.stem[0] in 'AD' and path.stem.split('_')[-1].isdigit():
                    _,start,stop=path.stem.split('_');expected=int(stop)-int(start)
                else:expected=39
                assert len(frames)==expected,(path,len(frames),expected)
            videos.append(dict(file=path.name,frames=expected,codec='h264',pix_fmt='yuv420p',fps=24,sha256=sha(path)))
        audit=dict(status='complete_full_decode',videos=videos,scope='codec validity, not model quality')
        (folder/'video_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        target=DEST/name;target.mkdir(parents=True,exist_ok=True)
        for file in folder.iterdir():
            if file.suffix in ('.json','.jpg','.mp4','.md'):
                shutil.copy2(file,target/file.name);assert sha(file)==sha(target/file.name)
        audits.append(dict(group=name,videos=len(videos)))
    report=dict(at=datetime.datetime.now().astimezone().isoformat(),rows=rows,local_reference_usable=False,
        reason='Changing current A/D barely affects outcome; same fixed history determines both directions',
        frozen_time_control_failed=True,stage1_accepted=False,video_audits=audits,
        review_scope='all39-frame static contact sheets reviewed for each A/D/reference and window12; not realtime playback',
        visual_observation='People and parking structure mostly remain recognizable; oracle resets at frames17/34 are not free-rollout continuity',
        geometry_launch=False,
        calibration_in_progress='Only native text/action time, with exact same weights/anchors/noise/window',
        evaluation_sha256={a:sha(BASE/f'positive_{a}/evaluation.json') for a in 'AD'})
    (BASE/'positive_review.json').write_text(json.dumps(report,indent=2)+'\n')
    table='\n'.join(f"| {r['reference']} | {r['chunk']} | {r['A']:+.6f} | {r['D']:+.6f} | {r['separation']:+.6f} |" for r in rows)
    text=f'''# 固定text/action clean time下的局部正控：未通过

两份Original-generated reference、每份三chunk各fork A/D、30steps全部完成。新T2保留Original directed attention、每个sigma重算可见窗口、无未来块输入，但沿用了原型的text/action clean-time覆盖。**不能把它无保留地称为原始H3推理函数。**

| 固定历史来源 | 当前chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
{table}

后续chunk里，参考历史A时两种当前动作都向正方向，参考历史D时两种当前动作都向负方向；当前A/D差很小。首块也未恢复方向。完整静态图中人物与停车场大体可辨，没有此前generated8的严重半透明分解；但动作门槛失败。39f串接在17/34帧恢复reference历史，有可见状态跳变，绝不能以它证明自由生成连续性。

扩大当前已知动作窗口到12latent/39RGB后，A=-0.745426、D=-0.767189，差约0.021764，仍同向。两条完整39f静态图中人物存在，动作几乎相同；窗口长度单独不能解释这次失效。它的控制粒度是39RGB，不是5latent在线原型。

源码追踪发现：发布H3-World的model_fn默认text/action time跟随video time(1−sigma)；原型fixed_prefix_timesteps=True将全部text设1。SolarWM Stage1确实采用clean text time，但其Stage0.5采用video time，需要经过训练适配。当前受控输入差异与attention mask是独立因素。新增5项CPU检查确认本30步网格上切换此flag只改变text/action时间，anchor/audio/history/video行时间不变；clean commit sigma0时anchor1 vs .999差异另记。

下一步已启动同权重/同anchor/同窗口的native text/action time对照，不先改anchor或加训练。原固定时间版本的geometry runner被正控门槛挡住，没有启动。旧几何结论仍是“在已固定的原型时间条件下改mask的相对影响”，不能直接据此声称完整复刻了原始H3动作函数。

[固定历史A局部视频](positive_A/AD_local_forks.mp4) · [固定历史D局部视频](positive_D/AD_local_forks.mp4) · [12latent窗口校准](window12_calibration/AD_window12.mp4)。全部27个MP4完整解码通过，原始指标与视频哈希保留。不是实时播放评审，没有新训练、没有Stage1验收通过。
'''
    (BASE/'FIXED_PREFIX_RESULTS.md').write_text(text)
    for name in ('positive_review.json','FIXED_PREFIX_RESULTS.md','report_fixed_reference.py','original_interval_flow.json'):
        shutil.copy2(BASE/name,DEST/name)
    print(json.dumps({'status':'completed_fixed_time_controls_failed_gate','rows':rows,'audits':audits},indent=2))


if __name__=='__main__':main()
