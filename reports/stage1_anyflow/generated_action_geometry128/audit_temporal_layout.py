"""Read-only audit of saved H3 RGB bins, action spans and retained RoPE."""
import json
import probe as p
p.setup('')
import torch
import abot_action
from causal.h3_cached import expand_packed_two_anchors,slice_packed,retime_anchor_position
from causal.train_online_selfrollout import action_condition

out=dict(status='running',rows=[],scope='index/layout audit, not image-space temporal responsiveness')
spans=abot_action.frame_spans(12)
for history in ('A','D'):
    c=torch.load(p.GENERATED/history/'conditioning.pt',map_location='cpu',weights_only=True)
    full=expand_packed_two_anchors(c['packed'],frame_rows=390)
    for chunk in (1,2):
        start=chunk*5;stop=min(start+5,12)
        local=slice_packed(full,start,stop,390)
        local=retime_anchor_position(local,full,anchor_rows=780,frame_index=start-1,
                                     frame_rows=390,anchor_slot=1)
        source=full['img_pos'][780+start*390:780+stop*390]
        assert torch.equal(local['img_position_ids'][0,local['img_pos'][780:]],
                           full['img_position_ids'][0,source])
        original_anchor_positions=full['img_position_ids'][0,full['img_pos'][:390]]
        assert torch.equal(local['img_position_ids'][0,local['img_pos'][:390]],original_anchor_positions)
        prior=full['img_pos'][780+(start-1)*390:780+start*390]
        assert torch.equal(local['img_position_ids'][0,local['img_pos'][390:780]],
                           full['img_position_ids'][0,prior])
        assert torch.equal(local['action_text_rows'],full['action_text_rows'])
        rows=[]
        for i in range(start,stop):
            lo,hi=map(int,full['action_text_rows'][i])
            assert (lo,hi)==tuple(full['action_text_spans_local'][i])
            rows.append(dict(latent_index=i,RGB_span=[int(x) for x in spans[i]],
                             action_text_span=[lo,hi]))
        out['rows'].append(dict(history=history,chunk=chunk,mapping=rows,
            original_anchor_position_unchanged=True,tail_anchor_global_frame=start-1,
            current_video_global_RoPE_unchanged=True,
            residual_A=action_condition('A',stop-start,'cpu',torch.float32)[0].tolist(),
            residual_D=action_condition('D',stop-start,'cpu',torch.float32)[0].tolist()))
out.update(status='passed',RGB_frames=39,latent_frames=12,
    chunk_RGB_spans=[[0,17],[17,34],[34,39]],
    limitation='Within-chunk video attention is bidirectional; frame-level strict causality is not claimed. Temporal VAE can spread visible changes across RGB boundaries.')
(p.OUT/'temporal_layout.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],chunk_RGB_spans=out['chunk_RGB_spans'])))
