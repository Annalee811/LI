"""Publish a separate, explicitly user-reviewed single-root correction."""
import base64,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_user_confirmed_new_axis_correction_20261029'
PREFIX='roma-results/0353-2/0353-2-B3/UNet-reviewed-new-root-0002-N1-v90'
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery'


def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else ''),'-H','Cache-Control: no-cache']
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)


def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    v=json.loads((OUT/'verification.json').read_text());assert v['passed'];assert v['other_root_pixels_changed']==0;assert not api('')['private']
    for p,h in v['immutable_input_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for t,h in v['prediction_sha256'].items():assert hashlib.sha256((OUT/'full'/(t+'_classes.png')).read_bytes()).hexdigest()==h
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    names=['RESULTS.md','completion.json','verification.json','correction_audit.json','Reviewed_connection_and_birth_supervision.json','evaluation_results.json','figure_manifest.json','Reviewed_root_0002-N1_support.png','T2_changed_pixels.png','Reviewed_root_instance_reassignment.csv','Dice_after_review_same_reference_V87.csv']
    for directory in ['full','ownership','source']:names += [str(p.relative_to(OUT)) for p in sorted((OUT/directory).rglob('*')) if p.is_file()]
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    for r in manifest['files']:
        p=OUT/'figures'/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];names+=['figures/'+p.name,'figures/'+p.stem+'_preview.png']
    cachep=OUT/'published_blob_cache.json';cache=json.loads(cachep.read_text()) if cachep.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            payload=dict(encoding='utf-8',content=data.decode('utf-8')) if name.endswith(('.json','.md','.py')) else dict(encoding='base64',content=base64.b64encode(data).decode())
            assert api('git/blobs',payload)['sha']==blob;assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachep,cache)
        entries.append(dict(path=PREFIX+'/'+name,mode='100644',type='blob',sha=blob));files.append(dict(path=PREFIX+'/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH+'?nonce='+str(time.time_ns()))['object']['sha'];tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Apply explicit user-reviewed T2 new-lateral0002-N1 correction; split false old-contact inheritance and preserve every other root, reference label and model weight'))['sha'];js(OUT/'publication_state.json',dict(parent=head,commit=commit));assert api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),'PATCH')['object']['sha']==commit
    tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    js(OUT/'download_links.json',dict(commit=commit,base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/',files=files,verification='GitHub API exact-byte readback and immutable committed-path verification.'))
    js(OUT/'publication_complete.json',dict(state='complete',commit=commit,files=len(files)));print('PUBLISHED',commit,flush=True)


if __name__=='__main__':main()
