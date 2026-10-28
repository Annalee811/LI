"""Publish actual root-axis prediction repairs without altering any older output path."""
import base64
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_root_state_prediction_repair_v88_20261028'
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery'
PREFIX='roma-results/0353-2/0353-2-B3/U-Net-root-axis-actual-prediction-repair-v88'


def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else ''),'-H','Cache-Control: no-cache']
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)


def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    verify=json.loads((OUT/'evaluation_verification.json').read_text());assert verify['passed'];assert not api('')['private']
    pv=json.loads((OUT/'prediction_verification.json').read_text())
    for p,h in pv['immutable_inputs_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for t,h in pv['prediction_sha256'].items():assert hashlib.sha256((OUT/'full'/(t+'_classes.png')).read_bytes()).hexdigest()==h
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    for r in manifest['files']:assert hashlib.sha256((OUT/'figures'/r['file']).read_bytes()).hexdigest()==r['sha256']
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    names=['RESULTS.md','completion.json','prediction_verification.json','prediction_freeze.json','evaluation_verification.json','evaluation_results.json','structural_audit.json','User_case_audit.json','figure_manifest.json','Known_cases_actual_repair_preview.png']
    names += [p.name for p in sorted(OUT.glob('*.csv'))]
    for directory in ['figures','full','changes','ownership','source']:
        names += [str(p.relative_to(OUT)) for p in sorted((OUT/directory).rglob('*')) if p.is_file()]
    cachepath=OUT/'published_blob_cache.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in dict.fromkeys(names):
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            payload=dict(encoding='utf-8',content=data.decode('utf-8')) if name.endswith(('.json','.md','.py')) else dict(encoding='base64',content=base64.b64encode(data).decode())
            assert api('git/blobs',payload)['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data
            cache[blob]=len(data);js(cachepath,cache)
        entries.append(dict(path=PREFIX+'/'+name,mode='100644',type='blob',sha=blob));files.append(dict(path=PREFIX+'/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH+'?nonce='+str(time.time_ns()))['object']['sha']
    tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha']
    commit=api('git/commits',dict(tree=tree,parents=[head],message='Repair real U-Net root-axis temporal predictions across all four fields; separate first-observed roots, old-root growth and independent root contacts; preserve annotations and weights with explicit residual-error audits'))['sha']
    js(OUT/'publication_state.json',dict(parent=head,commit=commit));assert api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),'PATCH')['object']['sha']==commit
    committed=api('git/trees/'+commit+'?recursive=1');assert not committed.get('truncated');paths={v['path']:v['sha'] for v in committed['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    js(OUT/'download_links.json',dict(commit=commit,base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/',files=files,verification='Exact GitHub API blob-byte readback and immutable commit-tree path verification.'))
    js(OUT/'publication_complete.json',dict(state='complete',commit=commit,files=len(files)));print('PUBLISHED',commit,flush=True)


if __name__=='__main__':main()
