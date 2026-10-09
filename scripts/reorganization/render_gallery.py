"""CPU-only, frame-exact H.264 gallery from saved videos; no model inference.

Sources and transformations are recorded per output. Existing MP4s are never
overwritten. Requires av, Pillow, numpy. All chosen sources are exactly 24fps.
"""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SUB = Path(__file__).resolve().parents[2]
AUDIT = SUB/'archive/reorganization_20261010'
GALLERY = SUB/'report/00_comparison_gallery'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
NAMES = ['V0_original','V1_native_causal','V2_rgb_anchor','V3_same_sigma']
TITLES = ['V0 Original H3-World','V1 Native causal | no new adapter',
          'V2 RGB + visual adaptation','V3 Same-sigma | C12->5']
CONFS = ['30 steps / full124 | Single I0 | bidirectional',
         '8 steps/chunk | latent dual | persistent CPU KV',
         '8 steps/chunk | RGB dual + tail16 | CPU KV',
         '30 steps/chunk | Single I0 | native time | T2']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def textfit(d, xy, text, size=18, color='white', width=602):
    font=ImageFont.truetype(FONT,size)
    while d.textbbox((0,0),text,font=font)[2] > width and size > 11:
        size-=1; font=ImageFont.truetype(FONT,size)
    assert d.textbbox((0,0),text,font=font)[2] <= width,text
    d.text(xy,text,font=font,fill=color)


def verify(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        meta=dict(codec=s.codec_context.name,pixel_format=s.codec_context.format.name,
                  fps=float(s.average_rate),width=s.width,height=s.height)
        pts=[Fraction(f.pts)*f.time_base for f in c.decode(video=0)]
    assert pts and pts[0]==0
    assert all(b-a==Fraction(1,24) for a,b in zip(pts,pts[1:])), path
    assert meta['codec']=='h264' and meta['pixel_format']=='yuv420p' and meta['fps']==24,path
    meta.update(frames=len(pts),duration_s=len(pts)/24,full_decode=True,
                constant_pts=True,sha256=sha(path),bytes=path.stat().st_size)
    return meta


def render(spec, assets):
    out=GALLERY/spec['filename']; assert not out.exists(),f'Refuse overwrite: {out}'
    flat=[assets[k] for row in spec['rows'] for k in row]
    cols=len(spec['rows'][0]); assert all(len(r)==cols for r in spec['rows'])
    count=spec['frames']; width=624*cols; rowheight=504; height=60+rowheight*len(spec['rows'])
    containers=[av.open(str(SUB/x['packaged_video'])) for x in flat]
    for c,x in zip(containers,flat):
        assert float(c.streams.video[0].average_rate)==24 and x['video']['frames']>=count
    iters=[c.decode(video=0) for c in containers]
    tmp=out.with_suffix('.partial.mp4')
    assert not tmp.exists()
    selected={0,min(38,count-1),min(39,count-1),count-1}
    previews=[]
    with av.open(str(tmp),'w',options={'movflags':'+faststart'}) as enc:
        stream=enc.add_stream('libx264',rate=24)
        stream.width,stream.height=width,height; stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'medium','threads':'2'}
        for i in range(count):
            canvas=Image.new('RGB',(width,height),(15,19,27)); d=ImageDraw.Draw(canvas)
            textfit(d,(12,5),spec['title'],20,width=width-24)
            textfit(d,(12,32),spec['caveat'],15,'#ffcf82',width-24)
            for j,(x,it) in enumerate(zip(flat,iters)):
                f=next(it); assert Fraction(f.pts)*f.time_base==Fraction(i,24)
                v=x['version']; px=j%cols*624; py=60+j//cols*rowheight
                textfit(d,(px+10,py+5),TITLES[v],21)
                textfit(d,(px+10,py+32),CONFS[v],17,'#c2d4ed')
                if v==3:
                    timing='Second-chunk sampling ~195s; startup reused'
                    cond='Self-generated history | recompute | NO persistent KV'
                else:
                    timing=f"Recorded full124 E2E {x['metrics']['wall_including_shared_setup_seconds']:.1f}s"
                    if count<124: timing+=' | NOT crop latency'
                    cond=['Released action LoRA | no causal training',
                          'Own action reads | fixed prefix time | feedback off',
                          'Causal action reads + feedback | trained visual QKV'][v]
                textfit(d,(px+10,py+55),cond,16,'#c2d4ed')
                textfit(d,(px+10,py+77),timing,16,'#c2d4ed')
                image=ImageOps.contain(f.to_image(),(624,360),Image.Resampling.LANCZOS)
                canvas.paste(image,(px+(624-image.width)//2,py+102+(360-image.height)//2))
                action=x['action']
                if v==3:
                    action=action.split('->')[0 if i<39 else 1]
                    phase='initial chunk [0,39)' if i<39 else 'second chunk [39,56)'
                    color='#71e2c1' if i<39 else '#ffd46b'
                    textfit(d,(px+10,py+465),f'Action {action} | {x["action"]} | {phase}',16,color)
                    if i>=39:
                        d.rectangle((px,py+101,px+623,py+462),outline='#ffd46b',width=3)
                else:
                    textfit(d,(px+10,py+465),f'Action {action} | source124; showing RGB[0,{count})',16)
                textfit(d,(px+10,py+486),f'RGB {i:03d} | {i/24:.3f}s | 24fps | seed13',13,'#a9b7cc')
            frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
            frame.pts=i; frame.time_base=Fraction(1,24)
            for packet in stream.encode(frame):enc.mux(packet)
            if i in selected:previews.append(canvas.resize((width//2,height//2)))
        for packet in stream.encode():enc.mux(packet)
    for c in containers:c.close()
    tmp.rename(out)
    meta=verify(out); assert meta['frames']==count
    sheet=Image.new('RGB',(previews[0].width*2,previews[0].height*((len(previews)+1)//2)),'white')
    for k,p in enumerate(previews):sheet.paste(p,((k%2)*p.width,(k//2)*p.height))
    preview=AUDIT/'contact_sheets'/f'{out.stem}.jpg';preview.parent.mkdir(exist_ok=True)
    sheet.save(preview,quality=90)
    receipt=dict(**spec,video=meta,renderer_sha256=sha(__file__),sources=[
        dict(id=x['id'],path=x['packaged_video'],sha256=x['sha256'],
             original_path=x['source'],frames_used=[0,count],source_frames=x['video']['frames']) for x in flat],
        transforms=['RGB frame-exact [0,N) selection','aspect-preserving Lanczos display resize + padding',
                    'labels + side-by-side','libx264 CRF18/yuv420p/24fps/faststart'],
        no_interpolation=True,no_loops=True,no_model_inference=True)
    (AUDIT/'renders'/f'{out.stem}.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(done=out.name,frames=count,bytes=meta['bytes'])),flush=True)
    return receipt


def main():
    GALLERY.mkdir(parents=True,exist_ok=True);(AUDIT/'renders').mkdir(exist_ok=True)
    assets={x['id']:x for x in json.loads((AUDIT/'selected_sources.json').read_text())}
    specs=[]
    def add(name,rows,n,title,caveat):
        specs.append(dict(filename=name+'.mp4',rows=rows,frames=n,title=title,caveat=caveat))
    add('V0_AD_reference',[['V0_A','V0_D']],124,'V0 action reference: A vs D',
        'Different actions intentionally; Original is a generated reference, not ground truth.')
    add('V0_WSAD_reference',[['V0_W','V0_S'],['V0_A','V0_D']],124,'V0 Original H3-World: W / S / A / D',
        '30 full-horizon steps | 124 RGB frames | released action LoRA | reference only')
    for v in (1,2,3):
        n=56 if v==3 else 124
        rows=[]
        caveat=('CROSS-PROTOCOL: same displayed range, different horizon/conditioning/history/topology; NOT an ablation.'
                if v==3 else 'Same scene/seed/actions; MULTI-FACTOR protocols (steps, conditioning, routing, adapters).')
        for a in 'AD':
            key=f'V{v}_{a+a if v==3 else a}'
            row=[f'V0_{a}',key]; rows.append(row)
            add(f'V{v}_{a}_vs_original',[row],n,f'Original H3-World vs V{v} | action {a}',caveat)
        add(f'V{v}_vs_original',rows,n,f'Original H3-World vs V{v} | A / D',caveat)
        if v>=2:
            rows=[]
            for a in 'AD':
                row=[f'V{v-1}_{a}',f'V{v}_{a+a if v==3 else a}']; rows.append(row)
                add(f'V{v}_{a}_vs_V{v-1}',[row],n,f'V{v-1} vs V{v} | action {a}',caveat)
            add(f'V{v}_vs_V{v-1}',rows,n,f'V{v-1} vs V{v} | A / D',caveat)
    add('V3_four_paths_56',[['V3_AA','V3_AD'],['V3_DA','V3_DD']],56,
        'V3 C12->5: A->A / A->D / D->A / D->D',
        'Boundary RGB39 | own generated history | only TWO chunks validated | NO GT reset / NO persistent KV')
    # Save complete recipe before any encoding.
    (AUDIT/'render_specs.json').write_text(json.dumps(specs,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda spec:render(spec,assets),specs))
    alias=GALLERY/'V1_vs_V0.mp4'; assert not alias.exists()
    shutil.copy2(GALLERY/'V1_vs_original.mp4',alias)
    results.append(dict(filename=alias.name,alias_of='V1_vs_original.mp4',video=verify(alias)))
    copies=[]
    for v in range(4):
        wanted=([f'V0_AD_reference.mp4',f'V0_WSAD_reference.mp4'] if v==0 else
                [f'V{v}_vs_original.mp4',f'V{v}_vs_V{v-1}.mp4']+([f'V3_four_paths_56.mp4'] if v==3 else []))
        for f in wanted:
            dst=SUB/'report'/NAMES[v]/'videos'/f;assert not dst.exists()
            shutil.copy2(GALLERY/f,dst)
            copies.append(dict(source=str((GALLERY/f).relative_to(SUB)),target=str(dst.relative_to(SUB)),sha256=sha(dst)))
    (AUDIT/'render_results.json').write_text(json.dumps(results,indent=2)+'\n')
    (AUDIT/'gallery_copies.json').write_text(json.dumps(copies,indent=2)+'\n')
    print(f'All {len(results)} gallery MP4s fully decoded; {len(copies)} portable version copies.')


if __name__=='__main__':main()
