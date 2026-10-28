"""Publish explicit verified GAT result files without modifying old models/editor."""
import base64,hashlib,json,subprocess,shutil
from pathlib import Path
from prepare import ROOT,OUT,MODELS,js
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/temporal-upload18-GAT-new-growth-v68'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)
def main():
    assert json.loads((OUT/'completion.json').read_text())['state']=='complete'
    assert json.loads((OUT/'evaluation_verification.json').read_text())['passed']
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    names=['README.md','RESULTS.md','Temporal_class_metrics_before_after.csv','Connection_metrics_GAT_control.csv','Class_continuity_comparison.csv','evaluation_verification.json','training_and_geometry_final_verification.json','figure_manifest.json','source_manifest.json','user_authorization.json','preflight_verification.json','identity_vs_class_target_verification.json']
    for row in manifest['files']:
        p=OUT/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        names.extend([row['file'],p.stem+'_preview.png'])
    for m in MODELS:
        for k in ['gat','mlp']:
            for n in ['best.pt','config.json','history.csv','completion.json','test_metrics.json']:names.append(m+'_'+k+'/'+n)
    if Path(__file__).resolve()!=(OUT/'source/publish.py').resolve():shutil.copy2(Path(__file__),OUT/'source/publish.py')
    for p in sorted((OUT/'source').glob('*.py')):names.append('source/'+p.name)
    assert not api('')['private']
    cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
        path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();statepath=OUT/'publication_state.json';previous=json.loads(statepath.read_text()) if statepath.exists() else {}
    if previous.get('commit')==head and previous.get('manifest_sha256')==signature:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha']
        commit=api('git/commits',dict(tree=tree,parents=[head],message='Report upload18 temporal GAT optimization with upload17 confirmed-ID connection supervision; preserve all mixed and negative results'))['sha'];js(statepath,dict(parent=head,commit=commit,manifest_sha256=signature));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    remote=api('git/trees/'+commit+'?recursive=1');assert not remote.get('truncated');paths={p['path']:p['sha'] for p in remote['tree'] if p['type']=='blob'};assert all(paths[v['path']]==v['blob'] for v in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    result=dict(commit=commit,verification='Public GitHub API byte readback and committed-path verification',archive='https://github.com/Annalee811/LI/tree/'+commit+'/'+PREFIX,report=base+'RESULTS.md',new_roots=base+'Fig_New_roots_before_after_GAT.tif',extension=base+'Fig_Tip_extension_before_after_GAT.tif',connections=base+'Fig_Connection_and_class_continuity.tif',class_metrics=base+'Temporal_class_metrics_before_after.csv',connection_metrics=base+'Connection_metrics_GAT_control.csv',continuity=base+'Class_continuity_comparison.csv',unet_comparison=base+'Fig_unet_four_period_before_after_GAT.tif',files=files)
    js(OUT/'download_links.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
