"""Certified frozen-prefix extension for EXP-005 C7/C8; CPU only.

Extend video and action temporal grids while preserving every old semantic
coordinate. This is an explicit extrapolation of the frozen action/video
lag, not a new call to H3's full-length packed builder. Both branches use
AA124 history; only action spans 37:47 differ between continue A and switch D.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP = HERE.parent
FROZEN = ROOT / 'H3-World/outputs/2026-10-09-22/chunk_partition_cb'
sys.path[:0] = [str(FROZEN/'runtime/code'),
               str(FROZEN/'runtime/DiffSynth-Studio-h3-v2'), str(EXP)]
from causal.local_topology import visible_inputs
from position_sw import assert_frozen_past_preserved, assert_native_video_grid, native_times


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+
                          t.view(torch.uint8).numpy().tobytes()).hexdigest()


def same_layout(a, b):
    for k in a:
        assert k in b, k
        if torch.is_tensor(a[k]): assert torch.equal(a[k], b[k]), k
        else: assert a[k] == b[k], k


def build_fixture(*, stop=47, extension_seed=130005):
    if stop != 47: raise ValueError('Only registered C7/C8 extension (47 latents)')
    torch.set_num_threads(4)
    paths = {x:FROZEN/f'source_coarse/inputs/parking_{x}.pt' for x in 'AD'}
    data = {x:torch.load(p,map_location='cpu',weights_only=True) for x,p in paths.items()}
    old = data['A']['packed']; same_layout(old, data['D']['packed'])
    source_text = {x:data[x]['pairs'][0]['prompts'][x] for x in 'AD'}
    spans = old['action_text_spans_local']; nold = len(spans)
    assert nold == 37 and len(old['text_pos']) == spans[-1][1]
    assert torch.equal(old['text_pos'], torch.arange(len(old['text_pos'])))
    rows = 390; added = stop - nold; old_text = len(old['text_pos'])
    lo, hi = spans[-1]; width = hi-lo
    assert width == 10 and len(old['img_pos']) == (1+nold)*rows
    for x in 'AD':
        for a,b in spans:
            assert b-a == width and torch.equal(source_text[x][a:b],source_text[x][lo:hi])
        assert torch.isfinite(source_text[x]).all()
    assert torch.equal(source_text['A'][:spans[0][0]], source_text['D'][:spans[0][0]])
    for name in ['anchor','audio_noise','initial_noise']:
        assert torch.equal(data['A'][name],data['D'][name]),name

    frozen, frozen_prompt = visible_inputs(old,source_text['A'],37,rows)
    delta = added*width; text_len = old_text+delta
    pold = int(frozen['action_video_start']); prefix = pold+delta
    seq_len = prefix+stop*rows
    coords = torch.empty((1,seq_len,3),dtype=old['img_position_ids'].dtype)
    coords[:,:old_text] = frozen['img_position_ids'][:,:old_text]
    coords[:,text_len:prefix] = frozen['img_position_ids'][:,old_text:pold]
    coords[:,prefix:prefix+nold*rows] = frozen['img_position_ids'][:,pold:]
    video_origin = float(frozen['img_position_ids'][0,pold,0])
    action_origin = float(frozen['img_position_ids'][0,spans[0][0],0])
    video_times = native_times(stop,video_origin)
    action_times = native_times(stop,action_origin)
    new_spans = list(spans)
    for j in range(nold,stop):
        a = old_text+(j-nold)*width; b=a+width;new_spans.append((a,b))
        coords[:,a:b] = frozen['img_position_ids'][:,lo:hi]
        coords[0,a:b,0] = action_times[j]
        a = prefix+j*rows; b=a+rows
        coords[:,a:b] = frozen['img_position_ids'][:,pold:pold+rows]
        coords[0,a:b,0] = video_times[j]
    tags = torch.empty(seq_len,dtype=torch.long)
    tags[:old_text] = frozen['token_tags'][:old_text]
    tags[old_text:text_len] = 1
    tags[text_len:prefix] = frozen['token_tags'][old_text:pold]
    tags[prefix:] = 0
    packed = dict(frozen)
    packed.update(img_position_ids=coords,token_tags=tags,
        text_pos=torch.arange(text_len),
        img_pos=torch.cat((frozen['img_pos'][:rows]+delta,torch.arange(prefix,seq_len))),
        audio_pos=frozen['audio_pos']+delta,
        action_text_spans_local=new_spans,
        action_text_rows=torch.tensor(new_spans,dtype=torch.long),
        action_video_start=prefix,seq_len=seq_len,action_real_used=seq_len,
        cu_seqlens=torch.tensor([0,seq_len],dtype=torch.int32))
    prompts={x:torch.cat((source_text['A'],source_text[x][lo:hi].repeat(added,1))) for x in 'AD'}
    old_noise=data['A']['initial_noise']
    generator=torch.Generator(device='cpu').manual_seed(extension_seed)
    tail=torch.randn((1,24,added,30,52),dtype=old_noise.dtype,generator=generator)
    noise=torch.cat((old_noise,tail),dim=2)
    assert torch.equal(noise[:,:,:nold],old_noise)
    assert_frozen_past_preserved(packed,frozen,frame_rows=rows)
    assert_native_video_grid(packed,frame_rows=rows,stop=stop,origin=video_origin)
    checks=[]
    for end in [12,17,22,27,32,37]:
        reference, ref_prompt=visible_inputs(old,source_text['A'],end,rows)
        for action in 'AD':
            candidate,text=visible_inputs(packed,prompts[action],end,rows)
            same_layout(reference,candidate)
            assert torch.equal(ref_prompt,text)
        checks.append(dict(stop=end,packed_and_prompt_exact=True))
    visible={}
    for end in [42,47]:
        lay,pa=visible_inputs(packed,prompts['A'],end,rows)
        lay_d,pd=visible_inputs(packed,prompts['D'],end,rows)
        same_layout(lay,lay_d)
        assert len(lay['action_text_rows'])==end
        assert torch.equal(pa[:old_text],frozen_prompt) and torch.equal(pd[:old_text],frozen_prompt)
        assert_native_video_grid(lay,frame_rows=rows,stop=end,origin=video_origin)
        visible[str(end)]=dict(prefix_rows=int(lay['action_video_start']),
            text_rows=len(pa),video_rows=end*rows,position_sha256=tensor_sha(lay['img_position_ids']),
            A_prompt_sha256=tensor_sha(pa),D_prompt_sha256=tensor_sha(pd))
    assert not torch.equal(prompts['A'][old_text:],prompts['D'][old_text:])
    actual_lags=[float(coords[0,prefix+j*rows,0]-coords[0,new_spans[j][0],0]) for j in range(stop)]
    assert max(abs(v-(video_origin-action_origin)) for v in actual_lags)<1e-10
    meta=dict(task='EXP-005/v2',certified_cpu=True,latent_stop=stop,
        source={x:dict(path=str(paths[x]),sha256=sha(paths[x])) for x in 'AD'},
        builder_sha256=sha(__file__),torch_version=str(torch.__version__),
        history='AA124 frozen EXP-003; actions A for latents 0:37 in both branches',
        current_actions={'A':'continue A from37','D':'switch to D from37'},
        embedding_extension='repeat saved final 10-row A/D span; exact equality verified for all37 source spans',
        video_origin=video_origin,action_origin=action_origin,action_video_lag=video_origin-action_origin,
        positional_extension='old coordinates byte-preserved; new video/action extend their original native grids with fixed lag',
        default_full_length_builder_equivalent=False,
        position_caveat='new action temporal coordinates exceed old I0 origin; do not rebase I0/audio/text or claim default full-length builder equivalence',
        noise_extension_seed=extension_seed,noise_generator='CPU torch.Generator, independent appended bfloat16 tensor',
        old_noise_sha256=tensor_sha(old_noise),extended_noise_sha256=tensor_sha(noise),tail_noise_sha256=tensor_sha(tail),
        anchor_sha256=tensor_sha(data['A']['anchor']),audio_sha256=tensor_sha(data['A']['audio_noise']),
        preeviction_regression=checks,visible=visible,gpu_forwards=0)
    fixture=dict(packed=packed,prompts=prompts,initial_noise=noise,anchor=data['A']['anchor'],
        audio_noise=data['A']['audio_noise'],frozen_reference=frozen,frozen_reference_prompt=frozen_prompt,
        metadata=meta)
    return fixture


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('fixture already frozen; no overwrite')
    fixture=build_fixture();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(fixture,args.output)
    report=dict(fixture['metadata'],fixture_path=str(args.output.resolve()),fixture_sha256=sha(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
