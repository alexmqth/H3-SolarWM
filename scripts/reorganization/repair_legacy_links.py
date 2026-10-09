"""Repair old Markdown paths, preserving original bytes and missing-resource facts."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
from validate import SUB, AUDIT, sha, targets

ROOT=SUB.parent


def main():
    before=json.loads((AUDIT/'legacy_link_issues.json').read_text())
    shutil.copy2(AUDIT/'legacy_link_issues.json',AUDIT/'legacy_link_issues_before.json')
    idx=json.loads((AUDIT/'file_index.json').read_text())
    originals=json.loads((AUDIT/'before_tracked.json').read_text())['files']
    byhash={}
    for x in originals:
        p=SUB/x['path']
        if p.exists() and p.suffix!='.md':byhash.setdefault(x['sha256'],p)
    changes=[];imports=[];actions=json.loads((AUDIT/'layout_actions.json').read_text())
    rawparents={}
    def locate(p,url):
        part=url.split('#',1)[0]
        for marker,base in [('submission/',SUB),('H3-World/',ROOT/'H3-World')]:
            if marker in part:
                c=base/part.split(marker,1)[1]
                if c.is_file() or c.is_dir():return c
        c=SUB/part
        if c.exists():return c
        if p in rawparents:
            c=(rawparents[p]/part).resolve()
            if c.exists():return c
        needle=part
        while needle.startswith('../'):needle=needle[3:]
        possible=[ROOT/x['path'] for x in idx if x['path'].endswith('/'+needle) and not x['path'].startswith('submission/')]
        possible=[p for p in possible if p.is_file()]
        if len(possible)==1:return possible[0]
        if possible and len({sha(p) for p in possible})==1:return possible[0]
        return None
    def package(src):
        if src.is_relative_to(SUB):return src
        h=sha(src)
        if h in byhash:return byhash[h]
        dst=SUB/'archive/referenced_assets'/src.relative_to(ROOT)
        assert src.stat().st_size<30*1024*1024,src
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        imports.append(dict(source=str(src.relative_to(ROOT)),target=str(dst.relative_to(SUB)),sha256=h,
                            bytes=src.stat().st_size))
        byhash[h]=dst
        if src.suffix=='.md':rawparents[dst]=src.parent
        return dst
    total=0
    for roundno in range(5):
        bad=[]
        for p in SUB.rglob('*.md'):
            if '.git' in p.parts:continue
            urls=[u for u in targets(p.read_text()) if not (p.parent/u).exists()]
            if urls:bad.append((p,set(urls)))
        if not bad:break
        for p,urls in bad:
            txt=p.read_text();original=txt;events=[]
            def replace(m):
                label,url=m.group(1),m.group(2)
                part=url.split('#',1)[0]
                if part not in urls:return m.group(0)
                src=locate(p,url)
                if src is None:
                    events.append(dict(old=url,new=None,status='MISSING_LEGACY_RESOURCE'))
                    return f'{label}（历史资源未找到/未打包：`{url}`）'
                dst=package(src);new=os.path.relpath(dst,p.parent)
                if '#' in url:new+='#'+url.split('#',1)[1]
                events.append(dict(old=url,new=new,status='path repaired; content unchanged'))
                return f'[{label}]({new})'
            txt=re.sub(r'\[([^\]]*)\]\(([^\s)]+)\)',replace,txt)
            if txt==original:continue
            raw=AUDIT/'original_link_documents'/p.relative_to(SUB)
            raw=raw.with_suffix('.txt');raw.parent.mkdir(parents=True,exist_ok=True)
            if not raw.exists():raw.write_bytes(p.read_bytes())
            p.write_text(txt)
            rel=str(p.relative_to(SUB))
            if rel in {x['path'] for x in originals} and rel not in {x['source'] for x in actions}:
                actions.append(dict(operation='repair Markdown paths only; conclusions unchanged',source=rel,target=rel,
                                    original_bytes=str(raw.relative_to(SUB))))
            changes.append(dict(document=rel,events=events,original_bytes=str(raw.relative_to(SUB))))
            total+=len(events)
    else:raise RuntimeError('Exceeded bounded link repair rounds')
    # Imported reports may have rebased links; original source bytes always preserved.
    for r in imports:r['packaged_sha256']=sha(SUB/r['target'])
    (AUDIT/'link_repairs.json').write_text(json.dumps(dict(repaired_occurrences=total,changes=changes,imports=imports),ensure_ascii=False,indent=2)+'\n')
    (AUDIT/'layout_actions.json').write_text(json.dumps(actions,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(occurrences=total,documents=len(changes),imported=len(imports),missing=sum(e['new'] is None for c in changes for e in c['events'])),ensure_ascii=False))


if __name__=='__main__':main()
