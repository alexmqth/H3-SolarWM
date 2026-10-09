"""CPU-only relabeling and comparisons for V2a/V2b; never overwrite MP4s."""
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import render_gallery as r

AUDIT=r.SUB/'archive/taxonomy_v2a_v2b_20261010'


def main():
    r.AUDIT=AUDIT
    r.TITLES=['V0 Original H3-World','V1 Native causal | no new adapter',
              'V2a RGB + visual adaptation','V2b Same-sigma | local bidir C12->5']
    (AUDIT/'renders').mkdir(exist_ok=True)
    assets={x['id']:x for x in json.loads((AUDIT/'selected_sources.json').read_text())}
    specs=[]
    def add(name,rows,n,title,caveat):specs.append(dict(filename=name+'.mp4',rows=rows,frames=n,title=title,caveat=caveat))
    for legacy,display,n in [(2,'V2a',124),(3,'V2b',56)]:
        for left,tag in [(0,'original'),(1,'V1')]:
            rows=[]
            caution=('CROSS-PROTOCOL: horizon, steps, conditioning, history and attention differ; NOT a single-variable ablation.'
                     if legacy==3 else 'Same scene/seed/actions; RGB + adapters + routing are a JOINT protocol, NOT an anchor-only ablation.')
            for action in 'AD':
                row=[f'V{left}_{action}',f'V{legacy}_{action*2 if legacy==3 else action}'];rows.append(row)
                add(f'{display}_{action}_vs_{tag}',[row],n,f'V{left} vs {display} | action {action}',caution)
            add(f'{display}_vs_{tag}',rows,n,f'V{left} vs {display} | A / D',caution)
    rows=[]
    for a in 'AD':
        row=[f'V2_{a}',f'V3_{a}{a}'];rows.append(row)
        add(f'V2a_vs_V2b_{a}',[row],56,f'V2a vs V2b | PARALLEL research approaches | action {a}',
            'CAPABILITY COMPARISON: 8 vs 30 steps/chunk; clean KV vs same-sigma recompute; NOT a single-variable ablation.')
    add('V2a_vs_V2b',rows,56,'V2a vs V2b | PARALLEL research approaches | A / D',
        'CAPABILITY COMPARISON: 8 vs 30 steps/chunk; clean KV vs same-sigma recompute; NOT a single-variable ablation.')
    add('V2b_four_paths_56',[['V3_AA','V3_AD'],['V3_DA','V3_DD']],56,
        'V2b local C12->5: A->A / A->D / D->A / D->D',
        'Boundary RGB39 | own history | only SECOND chunk validated | NO GT reset / NO persistent hidden KV')
    (AUDIT/'render_specs.json').write_text(json.dumps(specs,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda s:r.render(s,assets),specs))
    for x in results:x['taxonomy_renderer_sha256']=r.sha(__file__)
    (AUDIT/'render_results.json').write_text(json.dumps(results,indent=2)+'\n')
    copies=[]
    for d,names in {
      'V2a_rgb_anchor':['V2a_vs_original','V2a_vs_V1','V2a_vs_V2b'],
      'V2b_same_sigma_local_bidir':['V2b_vs_original','V2b_vs_V1','V2a_vs_V2b','V2b_four_paths_56']}.items():
        for name in names:
            src=r.GALLERY/(name+'.mp4');dst=r.SUB/'report'/d/'videos'/src.name
            assert not dst.exists();shutil.copy2(src,dst)
            copies.append(dict(source=str(src.relative_to(r.SUB)),target=str(dst.relative_to(r.SUB)),sha256=r.sha(dst)))
    (AUDIT/'gallery_copies.json').write_text(json.dumps(copies,indent=2)+'\n')
    print(f'Completed {len(results)} new labeled MP4s, {len(copies)} portable copies; no old MP4 overwritten.')


if __name__=='__main__':main()
