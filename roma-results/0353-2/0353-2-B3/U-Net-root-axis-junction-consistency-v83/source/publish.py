"""Publish verified full-field root consistency results; all historical files retained."""
import base64,hashlib,json,shutil,subprocess,time,gzip
from pathlib import Path
from prepare import ROOT,OUT,HERE,PERIODS,js,sha
D=OUT/'final';REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/U-Net-root-axis-junction-consistency-v83'
def api(path,obj=None,method=None):
 cmd=['gh','api',REPO+('/'+path if path else ''),'-H','Cache-Control: no-cache']
 if obj is not None:cmd+=['--input','-']
 if method:cmd+=['--method',method]
 p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[:1200])
 return json.loads(p.stdout)
def main():
 verify=json.loads((D/'evaluation_verification.json').read_text());assert verify['passed'];assert json.loads((D/'completion.json').read_text())['state']=='complete';assert not api('')['private'];manifest=json.loads((D/'figure_manifest.json').read_text());assert manifest['passed']
 for p in (D/'full').glob('*_classes.png'):assert sha(p)==verify['prediction_sha256'][p.name]
 (D/'source').mkdir(exist_ok=True);(D/'training').mkdir(exist_ok=True)
 for p in HERE.glob('*.py'):shutil.copy2(p,D/'source'/p.name)
 for name in ['joint_best.pt','joint_training_results.json','joint_training_history.csv','joint_gradient_verification.json','best_pair_head.pt','pair_training_results.json','pair_training_history.csv','preparation.json']:
  shutil.copy2(OUT/name,D/'training'/name)
 names=['RESULTS.md','DELIVERABLES.md','completion.json','evaluation_results.json','evaluation_verification.json','decoding_verification.json','figure_manifest.json','Figure_ROIs.json']+[p.name for p in sorted(D.glob('*.csv'))]
 names += ['source/'+p.name for p in sorted((D/'source').glob('*.py'))]+['training/'+p.name for p in sorted((D/'training').iterdir())]
 for row in manifest['files']:
  p=D/'figures'/row['file'];assert sha(p)==row['sha256'];names+=['figures/'+p.name,'figures/'+p.stem+'_preview.png']
 for t in PERIODS:names+=['full/'+t+'_classes.png','changes/'+t+'_edited.png','changes/'+t+'_allowed.png']
 # Keep the exact known regression comparison convenient to inspect.
 shutil.copy2(OUT/'final_case.png',D/'Known_case_before_after_preview.png');names.append('Known_case_before_after_preview.png')
 cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
 for name in dict.fromkeys(names):
  data=(D/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if cache.get(blob)!=len(data):
   assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob;assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
  path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=sha(D/name),blob=blob));print('VERIFIED',name,len(data),flush=True)
 head=api('git/ref/heads/'+BRANCH+'?nonce='+str(time.time_ns()))['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();sp=OUT/'publication_state.json';prev=json.loads(sp.read_text()) if sp.exists() else {}
 if prev.get('commit')==head and prev.get('signature')==signature:commit=head
 else:
  tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Train junction continuation and temporal root identity jointly; repair complete inconsistent root segments across all four fields with explicit residual-error audits'))['sha'];js(sp,dict(parent=head,commit=commit,signature=signature));updated=api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH');assert updated['object']['sha']==commit
 tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files);base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/';js(OUT/'download_links.json',dict(commit=commit,base=base,report=base+'RESULTS.md',index=base+'DELIVERABLES.md',metrics=base+'Pixel_class_metrics_reference20_and24.csv',remaining=base+'Remaining_third_type_junctions.csv',preview=base+'Known_case_before_after_preview.png',files=files,verification='GitHub API exact byte readback and committed path verification'));js(OUT/'publication_complete.json',dict(state='complete',commit=commit,files=len(files)));print('PUBLISHED',commit,flush=True)
if __name__=='__main__':main()
