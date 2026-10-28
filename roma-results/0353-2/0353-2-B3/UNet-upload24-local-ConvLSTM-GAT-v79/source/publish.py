"""Verify and publish only completed outputs; preserve all older result paths."""
import base64,hashlib,json,shutil,subprocess,time
from prepare import ROOT,OUT,ANCHOR,HERE,PERIODS,js,sha
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/UNet-upload24-local-ConvLSTM-GAT-v79'
def api(path,obj=None,method=None):
 cmd=['gh','api',REPO+('/'+path if path else ''),'-H','Cache-Control: no-cache']
 if obj is not None:cmd+=['--input','-']
 if method:cmd+=['--method',method]
 p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr[:1200])
 return json.loads(p.stdout)
def main():
 verify=json.loads((OUT/'evaluation_verification.json').read_text());assert verify['passed'] and verify['all_body_class_changes_outside_candidates_zero'];assert json.loads((OUT/'completion.json').read_text())['epochs_completed']==100;assert not api('')['private']
 for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
 names=['Temporal_evidence_proposals.csv','RESULTS.md','Three_temporal_classes_before_after.csv','Pixel_change_audit.csv','Known_root_identity_conflicts.csv','Root_connection_metrics.csv','evaluation_verification.json','preflight_verification.json','gradient_verification.json','preparation.json','config.json','completion.json','metrics.json','history.csv','epoch0_metrics.json','best.pt','figure_manifest.json','ROI_selection.json','data/source_progress_24.json'];names=[n for n in names if (OUT/n).exists()]
 names += ['source/'+p.name for p in sorted((OUT/'source').glob('*.py'))]
 for row in json.loads((OUT/'figure_manifest.json').read_text())['files']:
  p=OUT/'figures'/row['file'];assert sha(p)==row['sha256'];names += ['figures/'+p.name,'figures/'+p.stem+'_preview.png']
 for t in PERIODS:names += ['full/'+t+'_classes.png','frozen_unet/'+t+'_classes.png','changes/'+t+'_edited_pixels.png','candidates/'+t+'.npz']
 (OUT/'U_Net_stage1').mkdir(exist_ok=True)
 for n in ['history.csv','config.json','completion.json','best_temporal_head.pt','epoch0_metrics.json','test_metrics.json']:
  shutil.copy2(ANCHOR/'unet'/n,OUT/'U_Net_stage1'/n);names.append('U_Net_stage1/'+n)
 shutil.copytree(ANCHOR/'source',OUT/'U_Net_stage1/source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 names += [str(p.relative_to(OUT)) for p in sorted((OUT/'U_Net_stage1/source').rglob('*.py'))]
 cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
 for name in dict.fromkeys(names):
  data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if cache.get(blob)!=len(data):
   assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob;assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
  path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));files.append(dict(path=path,bytes=len(data),sha256=sha(OUT/name),blob=blob));print('VERIFIED',name,len(data),flush=True)
 head=api('git/ref/heads/'+BRANCH+'?nonce='+str(time.time_ns()))['object']['sha'];signature=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();sp=OUT/'publication_state.json';prev=json.loads(sp.read_text()) if sp.exists() else {}
 if prev.get('commit')==head and prev.get('signature')==signature:commit=head
 else:
  tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Train U-Net temporal head on revised24 labels, then freeze body classes for local-only ConvLSTM GAT junction corrections'))['sha'];js(sp,dict(parent=head,commit=commit,signature=signature));updated=api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH');assert updated['object']['sha']==commit
 tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');paths={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
 base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/';js(OUT/'download_links.json',dict(commit=commit,base=base,report=base+'RESULTS.md',metrics=base+'Three_temporal_classes_before_after.csv',files=files,verification='GitHub API exact byte readback and committed path verification'));js(OUT/'publication_complete.json',dict(state='complete',commit=commit,files=len(files)));print('PUBLISHED',commit,flush=True)
if __name__=='__main__':main()
