"""Publish verified U-Net comparison artifacts, preserving all historical paths."""
import base64,hashlib,json,shutil,subprocess
from build import ROOT,OUT,HERE,js,sha
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/UNet-ConvLSTM-GAT-comparison-v75'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)
def main():
    verify=json.loads((OUT/'verification.json').read_text());assert verify['passed'];assert not api('')['private']
    (OUT/'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    names=['RESULTS.md','UNet_before_after_latest20_metrics.csv','metrics.json','verification.json']
    for r in verify['figures']:
        p=OUT/r['file'];assert sha(p)==r['sha256'];names += [p.name,p.stem+'_preview.png']
    names += ['source/'+p.name for p in sorted((OUT/'source').glob('*.py'))]
    cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name in names:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
            assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
        path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=sha(OUT/name),blob=blob));print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();sp=OUT/'publication_state.json';prev=json.loads(sp.read_text()) if sp.exists() else {}
    if prev.get('commit')==head and prev.get('signature')==signature:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Publish U-Net ConvLSTM GAT100-epoch comparison and same-reference control; retain both improvements and regressions'))['sha'];js(sp,dict(parent=head,commit=commit,signature=signature));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    links=dict(commit=commit,base=base,report=base+'RESULTS.md',metrics=base+'UNet_before_after_latest20_metrics.csv',files=files,verification='GitHub API exact byte readback and committed path verification');js(OUT/'download_links.json',links);print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
