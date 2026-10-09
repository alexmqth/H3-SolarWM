"""Same-RGB-interval action comparison without re-encoding or new metrics."""
from pathlib import Path
from datetime import datetime
import hashlib
import itertools
import json
import sys

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
sys.path.insert(0,str(RT/'code/causal'))
import evaluate_action_control as flow


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original_frames=flow._frames
    def first17(path):return itertools.islice(original_frames(path),17)
    rows=[]
    try:
        for name in ('matched_first5','coarse_A','coarse_D'):
            receipt=json.loads((BASE/name/'evaluation.json').read_text())
            record=[v for v in receipt['records'] if v['window']==0]
            assert len(record)==2 and all('flow' in v for v in record)
            for action in 'AD':
                path=BASE/name/f'window0_{action}.mp4'
                row=next(v for v in record if v['action']==action)
                assert sha(path)==row['video_sha256']
                flow._frames=first17
                bounded=flow.evaluate(path)
                if name=='matched_first5':assert bounded['horizontal_flow_px']==row['flow']['horizontal_flow_px']
                rows.append(dict(group=name,action=action,source_sha256=sha(path),
                    first17_flow=bounded['horizontal_flow_px']['mean'],
                    whole_window_flow=row['flow']['horizontal_flow_px']['mean'],
                    whole_window_frames=row['flow']['frames'],first17_frames=bounded['frames']))
    finally:flow._frames=original_frames
    result=dict(at=datetime.now().astimezone().isoformat(),records=rows,
        evaluator_sha256=sha(RT/'code/causal/evaluate_action_control.py'),
        method='Existing Farneback evaluator with input-frame iterator limited to first17; no reencoding',
        interpretation='Same first17 displayed RGB interval;12-latent generator/decoder still uses its entire known39RGB window. Not an equal interactive-latency comparison or proof of later-window ability.',stage1_accepted=False)
    (BASE/'first_window_interval_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f"| {r['group']} | {r['action']} | {r['first17_flow']:+.6f} | {r['whole_window_flow']:+.6f} | {r['whole_window_frames']} |" for r in rows)
    report=f'''# 同输入首窗口：12latent恢复响应，5latent仍失败

本页只报告已完成的首窗口，不把运行中的后续窗口写成通过。Original权重、native时间、单I0、30步、full37布局与initial noise相同；第一5latent噪声为同一张量的相同slice。

| 方法/历史标签 | 动作 | 前17RGB flow | 整个当前窗口flow | 当前窗口RGB帧数 |
|---|---|---:|---:|---:|
{table}

完整窗口的首5latent A−D约0.005，首12latent A−D约2.037。为避免只因39帧比17帧长而误判，另列相同前17帧的原Farneback指标；直接截断解码帧iterator，不二次压缩，原17帧指标逐值重放一致。

12latent可见当前窗口包含39帧，前17RGB也可受该窗口后部latent影响；它的控制/显示粒度与5latent不同，不能据此宣称低延迟已经改善。全部17/39静态图中人物和停车场可辨，12latent的A会转向并产生不同运动，5latent的A/D近同。静态图评审不是实时播放评审。后续两个窗口、同状态velocity geometry和GT局部画面仍待验证。

视频：[5latent A/D](matched_first5/AD_local_forks.mp4)；[12latent A](first_window_snapshot/coarse_A/window0_A.mp4)、[12latent D](first_window_snapshot/coarse_A/window0_D.mp4)。原评测完整receipt写在输出目录，提交包中的first_window_snapshot明确仅为首窗口快照。
'''
    (BASE/'FIRST_WINDOW_RESULTS.md').write_text(report)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
