import base64,hashlib,json,shutil,subprocess
from pathlib import Path
from evaluate import ROOT,OUT,HERE,js
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/root-three-class-evaluation-v72'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)
def main():
    assert json.loads((OUT/'verification.json').read_text())['passed'];assert not api('')['private']
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    figures=json.loads((OUT/'figure_manifest.json').read_text())['files']
    names=['RESULTS.md','Three_classes_connection_metrics.csv','Three_classes_connection_by_period.csv','GAT_three_classes_summary.csv','Three_classes_original_temporal_baseline.csv','verification.json','figure_manifest.json']
    for row in figures:
        p=OUT/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];names += [p.name,p.stem+'_preview.png']
    names += ['source/'+p.name for p in sorted((OUT/'source').glob('*.py'))]
    cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
        path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();statepath=OUT/'publication_state.json';prev=json.loads(statepath.read_text()) if statepath.exists() else {}
    if prev.get('commit')==head and prev.get('signature')==signature:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Separately evaluate new, extending and persistent reference segments; distinguish within-period connection from cross-period identity continuation'))['sha'];js(statepath,dict(parent=head,commit=commit,signature=signature));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    links=dict(commit=commit,report=base+'RESULTS.md',metrics=base+'Three_classes_connection_by_period.csv',pooled_connection_metrics=base+'Three_classes_connection_metrics.csv',frozen_temporal_baseline=base+'Three_classes_original_temporal_baseline.csv',new_figure=base+'Fig_newly_observed_connection_evaluation.tif',extension_figure=base+'Fig_tip_extension_connection_evaluation.tif',persistent_figure=base+'Fig_persistent_connection_evaluation.tif',verification='GitHub API byte readback and committed path verification',files=files)
    js(OUT/'download_links.json',links);js(OUT/'completion.json',dict(state='complete',categories=3,temporal_pixel_retraining=False,commit=commit));print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
