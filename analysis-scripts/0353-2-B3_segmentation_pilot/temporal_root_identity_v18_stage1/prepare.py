"""Conservative, logged semantic corrections from U-Net + preceding manual evidence."""
import argparse,copy,json,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();r=a.run;d=r/'data';assert not d.exists();d.mkdir(parents=True)
for n in ['upper','original','frozen','revised','v14_reference','changed','patches']:(d/n).mkdir()
for n in ['original_human_progress.json','paired_manifest.csv','spatial_manifest.csv','dataset_summary.json']:shutil.copy2(a.previous/'data'/n,d/n)
shutil.copy2(a.previous/'data/active_progress.json',d/'previous_v17_progress.json');project=json.loads((d/'previous_v17_progress.json').read_text());log=[];summaries=[]
for ti,(t,item) in enumerate(zip(c.PERIODS,project['images'])):
 for src,dst in [('upper','upper'),('original','original'),('revised','frozen'),('v14_reference','v14_reference')]:
  for pth in (a.previous/'data'/src).glob(t+'_*'):shutil.copy2(pth,d/dst/pth.name)
 gt=np.array(Image.open(d/'frozen'/f'{t}_classes.png'));revised=gt.copy();gate=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0;changes=np.zeros_like(gt,bool)
 if ti:
  prev=c.PERIODS[ti-1];previous_human=np.array(Image.open(d/'original'/f'{prev}_classes.png'));human_root=(previous_human>0)&(previous_human<255);hd=ndi.distance_transform_edt(~human_root);axis=c.load_axes(r/'teacher_axis_cache'/f'{t}.npz');meta=json.loads((r/'teacher_axis_cache'/f'{t}.json').read_text());matches=np.load(r/'teacher_axis_cache'/f'{t}_matches.npz');proposal=np.array(Image.open(r/'teacher_identity_preview'/f'{t}_temporal_classes.png'));teacher=np.load(a.previous/'unet/four_period_prediction'/f'{t}_class_probability.npz')['probability'].astype('float32');binary=np.load(a.roots/'unet/upper'/f'{t}_probability.npy')
  for ai,path in enumerate(axis['paths'],1):
   if len(path)<32:continue
   xy=axis['xy'][path];yy,xx=xy.T;owned=axis['axis_map'][yy,xx]==ai;visible=matches['node_score'][path]>.4;u_fraction=float(visible.mean());h_fraction=float((hd[yy,xx]<=4).mean());joint=float((visible&(hd[yy,xx]<=4)).mean());absence=float((hd[yy,xx]>6).mean());node_probs=teacher[:,yy,xx].mean(1);root_score=float(binary[yy,xx].mean());region=axis['axis_map']==ai;oldvals=gt[region];typed=(oldvals>0)&(oldvals<255);counts=np.array([(oldvals==k).sum() for k in [1,2,3]]);reason=None;eligible=np.zeros_like(gt,bool)
   if root_score>=.8 and u_fraction>=.80 and joint>=.70:
    # Change only pixels assigned to this axis and inside a well observed old core.
    sm=ndi.median_filter((visible&(hd[yy,xx]<=4)).astype('float32'),size=min(7,2*(len(path)//2)+1),mode='nearest');supported=np.flatnonzero(sm>.5)
    if len(supported)>=24:
     goodpath=path[supported[0]:supported[-1]+1];py,px=axis['xy'][goodpath].T
     ry,rx=np.where(region);_,nn=cKDTree(axis['xy'][path]).query(np.stack([ry,rx],1));in_core=(nn>=supported[0])&(nn<=supported[-1]);eligible[ry[in_core],rx[in_core]]=True;eligible&=(gt>0)&(gt<255)&(gt!=1);reason='Persistent old core supported by both preceding manual root and U-Net root correspondence';revised[eligible]=1
   elif root_score>=.8 and u_fraction<=.10 and absence>=.90 and float(node_probs[1:].sum())>=.80:
    # Root was not observed in either preceding manual traces or U-Net support;
    # without an old-tip continuation it is newly observed, not elongation.
    rrs=json.loads((r/'teacher_identity_preview'/f'{t}_root_axes.json').read_text());assert len(rrs)==len(axis['paths']);rr=rrs[ai-1]
    if rr['mode']=='newly_observed_whole_root' and not rr['previous_tip_continuation']:
     eligible=region&(gt>0)&(gt<255)&(gt!=2);reason='Entire newly observed root: absent from preceding manual and U-Net root support; no old-tip continuation';revised[eligible]=2
   if eligible.any():
    ey,ex=np.where(eligible);transition={f'{old}->{new}':int(((gt==old)&(revised==new)&eligible).sum()) for old in [1,2,3] for new in [1,2,3] if old!=new and ((gt==old)&(revised==new)&eligible).any()};log.append(dict(period=t,root_axis_id=ai,changed_pixels=int(eligible.sum()),bbox_xyxy=[int(ex.min()),int(ey.min()),int(ex.max()+1),int(ey.max()+1)],transitions=transition,reason=reason,previous_U_Net_visibility=u_fraction,previous_manual_visibility=h_fraction,joint_previous_visibility=joint,previous_manual_absence=absence,current_binary_U_Net_score=root_score,U_Net_temporal_scores_P_N_G=node_probs.tolist(),human_reviewed=False,teacher_geometry_source_SHA256=c.sha(r/'teacher_axis_cache'/f'{t}.npz')));changes|=eligible
 # Preserve root geometry and unknown labels. T1 remains all baseline.
 assert np.array_equal(revised>0,gate);assert np.array_equal(revised==255,gt==255)
 if t=='T1':assert np.array_equal(revised,gt)
 Image.fromarray(revised).save(d/'revised'/f'{t}_classes.png');Image.fromarray(gate.astype('uint8')*255).save(d/'revised'/f'{t}_root_mask.png');Image.fromarray(changes.astype('uint8')*255).save(d/'changed'/f'{t}_semantic_changes.png');raw=np.zeros_like(gt)
 for cls,ui in [(1,3),(2,1),(3,2),(4,4)]:raw[revised==cls]=ui
 item['layers']['temporal']=c.pack(raw);item['reviewed']=False;summaries.append(dict(period=t,semantic_pixels_changed=int(changes.sum()),root_pixels=int(gate.sum()),untyped_pixels=int((revised==255).sum()),root_geometry_unchanged=True))
for row in c.rows(d/'spatial_manifest.csv'):
 x,y=int(row['x']),int(row['y']);out=d/'patches'/row['split'];out.mkdir(exist_ok=True)
 for t in c.PERIODS:
  for ref in ['original','frozen','revised']:Image.open(d/ref/f'{t}_classes.png').crop((x,y,x+256,y+256)).save(out/f'{row["tile_id"]}_{t}_{ref}_classes.png')
  Image.open(d/'upper'/f'{t}_RGB.png').crop((x,y,x+256,y+256)).save(out/f'{row["tile_id"]}_{t}_RGB.png')
meta=dict(version='temporal_root_identity_v18',position_id='0353-2-B3',reference_type='Conservative U-Net/preceding-human-supported temporal corrections, pending user review',original_human_SHA256=c.sha(d/'original_human_progress.json'),previous_V17_progress_SHA256=c.sha(d/'previous_v17_progress.json'),root_geometry_unchanged=True,RGB_unchanged=True,semantic_replacement_is_selective_not_whole_field=True,semantic_correction_rules='Persistent old core requires joint prior manual and U-Net support; NEW correction requires absence in both plus current model novel-root evidence and no old-tip continuation',originals_retained=True,unknown_labels_preserved=True,validation_selection_reference='Frozen unchanged V17 manual-temporal/U-Net-morphology reference',corrected_reference_is_not_independent_human_truth=True,independent_external_test_images=0,death_included=False)
project['annotation_revision']=meta;c.js(d/'active_progress.json',project);c.js(d/'revision_provenance.json',meta);c.js(d/'revision_log.json',log);c.js(d/'revision_summary.json',summaries)
source=r/'source_RGB/0353-2-B3';source.mkdir(parents=True)
for t in c.PERIODS:os.link(d/'upper'/f'{t}_RGB.png',source/f'{t}_RGB.png')
print('CONSERVATIVE_SEMANTIC_REVISION',summaries,'events',len(log),flush=True)
