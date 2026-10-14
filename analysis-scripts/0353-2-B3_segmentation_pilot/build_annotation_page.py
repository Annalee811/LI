#!/usr/bin/env python3
"""Embed the exact PNG bytes for the fixed study into an offline annotation page."""
from pathlib import Path
import argparse,base64,csv,hashlib,json

p=argparse.ArgumentParser()
p.add_argument('--kit',type=Path,required=True)
p.add_argument('--template',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
with (a.kit/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==60 and len({r['image_id'] for r in rows})==60
data=[];expected={}
for row in rows:
    file=a.kit/row['image_path'];b=file.read_bytes()
    assert hashlib.sha256(b).hexdigest()==row['image_sha256']
    assert file.name==row['image_id']+'.png'
    data.append(dict(name=file.name,base64=base64.b64encode(b).decode()))
    expected[file.name]=row['image_sha256']
page=a.template.read_text(encoding='utf-8')
for ident,value in [('embedded-dataset',data),('expected-images',expected)]:
    original=f'<script id="{ident}" type="application/json">'+('[]' if isinstance(value,list) else '{}')+'</script>'
    assert original in page
    page=page.replace(original,f'<script id="{ident}" type="application/json">'+json.dumps(value,separators=(',',':'))+'</script>')
a.output.write_text(page,encoding='utf-8')
print('SELF_CONTAINED_ANNOTATION_PAGE_READY',len(rows),a.output.stat().st_size)
