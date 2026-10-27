import base64,hashlib,json,subprocess,shutil
from pathlib import Path
R=Path('/workspace/LI');O=R/'results/B3_temporal_local_window_v64_20261027';HERE=Path(__file__).resolve().parent
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/temporal-local-new-extension-v64'
def api(path,obj=None,method=None):
 command=['gh','api',REPO+('/'+path if path else '')]
 if obj is not None:command+=['--input','-']
 if method:command+=['--method',method]
 r=subprocess.run(command,cwd=R,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[:1200])
 return json.loads(r.stdout)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 report=json.loads((O/'verification.json').read_text());assert report['passed'];figure=O/report['output']['file'];assert hashlib.sha256(figure.read_bytes()).hexdigest()==report['output']['SHA256'];assert not api('')['private']
 for name in ['render.py','select_window.py','publish.py']:shutil.copy2(HERE/name,O/'source'/name)
 names=[figure.name,'ROI_selection.json','verification.json','class_counts.csv','README.md','source/render.py','source/select_window.py','source/publish.py']
 cachepath=O/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
 for name in names:
  data=(O/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if cache.get(blob)!=len(data):
   assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
   assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);write(cachepath,cache)
  dest=PREFIX+'/'+name;entries.append(dict(path=dest,mode='100644',type='blob',sha=blob));files.append(dict(path=dest,bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',name,len(data),flush=True)
 head=api('git/ref/heads/'+BRANCH)['object']['sha'];manifest=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();statepath=O/'publication_state.json';prior=json.loads(statepath.read_text()) if statepath.exists() else {}
 if prior.get('commit')==head and prior.get('manifest_SHA256')==manifest:commit=head
 else:
  tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(message='Replace temporal illustrative detail window; show new roots and tip extension in T2-T4 without changing predictions',tree=tree,parents=[head]))['sha'];write(statepath,dict(commit=commit,parent=head,manifest_SHA256=manifest));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
 assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit;tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[f['path']]==f['blob'] for f in files)
 links=dict(commit=commit,figure_url='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'+figure.name,verification='Public GitHub API byte read-back and committed path verification',files=files);write(O/'download_links.json',links);print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
