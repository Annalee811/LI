"""Publish only verified U-Net preview artifacts; preserve all old branch files."""
import base64,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_temporal_upload18_UNet_preview_v67_20261028';HERE=Path(__file__).resolve().parent
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/temporal-upload18-unet-preview-v67'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    verify=json.loads((OUT/'figure_verification.json').read_text());assert verify['passed']
    for row in verify['outputs']:assert hashlib.sha256((OUT/row['file']).read_bytes()).hexdigest()==row['sha256']
    assert not api('')['private']
    (OUT/'source').mkdir(exist_ok=True)
    for name in ['infer.py','render.py','publish.py']:shutil.copy2(HERE/name,OUT/'source'/name)
    names=['README.md','inference_verification.json','figure_verification.json','UNet_history_0_to_100.csv','UNet_internal_holdout_metrics.json']
    for stem in ['Fig1_UNet_four_period_temporal_comparison','Fig2_UNet_local_temporal_detail','Fig3_UNet_100_epoch_training_curves']:
        names += [stem+'.tif',stem+'_preview.png']
    names += ['source/infer.py','source/render.py','source/publish.py']
    cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data
            cache[blob]=len(data);js(cachepath,cache)
        dest=PREFIX+'/'+name;entries.append(dict(path=dest,mode='100644',type='blob',sha=blob));files.append(dict(path=dest,bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),blob=blob))
        print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];manifest=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    statepath=OUT/'publication_state.json';prior=json.loads(statepath.read_text()) if statepath.exists() else {}
    if prior.get('commit')==head and prior.get('manifest_SHA256')==manifest:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha']
        commit=api('git/commits',dict(message='Preview completed upload18 U-Net temporal head: frozen root geometry, true 100-epoch curves and four-period qualitative results',tree=tree,parents=[head]))['sha']
        js(statepath,dict(commit=commit,parent=head,manifest_SHA256=manifest))
        api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated')
    paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[v['path']]==v['blob'] for v in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    links=dict(commit=commit,verification='GitHub API byte read-back and committed-path verification',
        full_figure_url=base+'Fig1_UNet_four_period_temporal_comparison.tif',full_preview_url=base+'Fig1_UNet_four_period_temporal_comparison_preview.png',
        detail_figure_url=base+'Fig2_UNet_local_temporal_detail.tif',detail_preview_url=base+'Fig2_UNet_local_temporal_detail_preview.png',
        training_figure_url=base+'Fig3_UNet_100_epoch_training_curves.tif',training_preview_url=base+'Fig3_UNet_100_epoch_training_curves_preview.png',files=files)
    js(OUT/'download_links.json',links);js(OUT/'completion.json',dict(state='complete',figures=3,unet_training_epochs=100,selected_epoch=37,GAT=False,commit=commit))
    print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
