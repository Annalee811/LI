"""Publish the completed, verified three-model temporal experiment without overwrites."""
import base64,hashlib,json,shutil,subprocess
from pathlib import Path
from prepare import ROOT,OUT,HERE,MODELS,PERIODS,js,sha
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/temporal-ConvLSTM-GAT-upload20-v74'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)
def main():
    assert json.loads((OUT/'completion.json').read_text())['state']=='complete';assert json.loads((OUT/'evaluation_verification.json').read_text())['passed'];assert not api('')['private']
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    # Preserve the launch source manifest and add a separate final postprocessing snapshot.
    js(OUT/'final_source_manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted((OUT/'source').glob('*.py'))])
    names=['RESULTS.md','DELIVERABLES.md','completion.json','Three_classes_accuracy_before_after.csv','Three_model_macro_summary.csv','Connection_accuracy_within_and_across.csv','Connection_accuracy_by_class.csv','Root_temporal_consistency.csv','evaluation_verification.json','preflight_verification.json','figure_manifest.json','baseline_preserved_figures_complete.json','preparation.json','source_manifest.json','final_source_manifest.json','ROI_selection.json','data/source_progress_20.json']
    names += ['source/'+p.name for p in sorted((OUT/'source').glob('*.py'))]
    for row in json.loads((OUT/'figure_manifest.json').read_text())['files']:
        p=OUT/'figures'/row['file'];assert sha(p)==row['sha256'];names += ['figures/'+p.name,'figures/'+p.stem+'_preview.png']
    for row in json.loads((OUT/'baseline_preserved_figures_complete.json').read_text())['original_figures']:
        assert sha(OUT/row['file'])==row['sha256'];names.append(row['file'])
    for m in MODELS:
        for v in ['control','convlstm_gat']:
            base=m+'_'+v;p=OUT/base;done=json.loads((p/'completion.json').read_text());assert done['epochs_completed']==100 and sha(p/'best.pt')==done['checkpoint_sha256']
            names += [base+'/'+n for n in ['completion.json','config.json','history.csv','epoch0_metrics.json','best.pt','latest20_metrics.json','root_period_diagnostics.csv'] if (p/n).exists()]
            names += [base+'/full/'+t+'_classes.png' for t in PERIODS]
        names += ['baseline_original/'+m+'/history.csv','baseline_original/'+m+'/completion.json',m+'_baseline/latest20_metrics.json']
    names=list(dict.fromkeys(names));cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
        path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=sha(OUT/name),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();sp=OUT/'publication_state.json';prev=json.loads(sp.read_text()) if sp.exists() else {}
    if prev.get('commit')==head and prev.get('signature')==signature:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Preserve original temporal baselines and publish latest20 ConvLSTM GAT100-epoch joint/control comparisons for three models'))['sha'];js(sp,dict(parent=head,commit=commit,signature=signature));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    links=dict(commit=commit,base=base,report=base+'RESULTS.md',index=base+'DELIVERABLES.md',metrics=base+'Three_classes_accuracy_before_after.csv',summary=base+'Three_model_macro_summary.csv',verification='GitHub API byte readback and committed path verification',files=files);js(OUT/'download_links.json',links);js(OUT/'publication_complete.json',dict(commit=commit,files=len(files),state='complete'));print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
