"""Publish a separate immutable report without replacing previous predictions/results."""
import base64, hashlib, json, subprocess, time
from pathlib import Path

ROOT=Path('/workspace/LI')
OUT=ROOT/'results/B3_V90_manual_original_UNet_report_20261029'
PREFIX='roma-results/0353-2/0353-2-B3/V90-manual-original-UNet-comparison-and-original-100-epoch-curves'
REPO='repos/Annalee811/LI'; BRANCH='annotation-restore-v50-delivery'


def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else ''),'-H','Cache-Control: no-cache']
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)


def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    v=json.loads((OUT/'verification.json').read_text()); assert v['passed']
    assert json.loads((OUT/'completion.json').read_text())['state']=='complete'
    assert not api('')['private']
    for p,h in v['immutable_input_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    entries=[];files=[]
    rows=json.loads((OUT/'artifact_manifest.json').read_text())
    rows.append(dict(file='artifact_manifest.json',sha256=hashlib.sha256((OUT/'artifact_manifest.json').read_bytes()).hexdigest()))
    cachep=OUT/'published_blob_cache.json';cache=json.loads(cachep.read_text()) if cachep.exists() else {}
    for row in rows:
        name=row['file'];data=(OUT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            payload=dict(encoding='utf-8',content=data.decode('utf-8')) if name.endswith(('.json','.md','.py','.html','.svg','.csv')) else dict(encoding='base64',content=base64.b64encode(data).decode())
            assert api('git/blobs',payload)['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data
            cache[blob]=len(data);js(cachep,cache)
        entries.append(dict(path=PREFIX+'/'+name,mode='100644',type='blob',sha=blob))
        files.append(dict(path=PREFIX+'/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob))
        print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH+'?nonce='+str(time.time_ns()))['object']['sha']
    tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha']
    commit=api('git/commits',dict(tree=tree,parents=[head],message='Export immutable V90/manual/original-U-Net four-period and local comparisons, same-reference accuracy/Dice, and exact original three-model 100-epoch train/validation loss curves'))['sha']
    js(OUT/'publication_state.json',dict(parent=head,commit=commit))
    assert api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),'PATCH')['object']['sha']==commit
    committed=api('git/trees/'+commit+'?recursive=1');assert not committed.get('truncated')
    paths={r['path']:r['sha'] for r in committed['tree'] if r['type']=='blob'}
    assert all(paths[r['path']]==r['blob'] for r in files)
    js(OUT/'download_links.json',dict(commit=commit,base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/',files=files,verification='GitHub API exact-byte blob readback and immutable committed-path verification; raw host not tested in this runtime.'))
    js(OUT/'publication_complete.json',dict(state='complete',commit=commit,files=len(files)))
    print('PUBLISHED',commit,flush=True)


if __name__=='__main__':main()
