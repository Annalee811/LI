"""Keep manual semantics; fit width/extent to U-Net and flag unresolved new pixels."""
import argparse,copy,json,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;d=r/'data';assert not d.exists();d.mkdir(parents=True)
for n in ['upper','original','revised','v14_reference','changed','patches']:(d/n).mkdir()
for n in ['original_human_progress.json','paired_manifest.csv','spatial_manifest.csv','dataset_summary.json']:shutil.copy2(a.previous/'data'/n,d/n)
original=json.loads((d/'original_human_progress.json').read_text());project=copy.deepcopy(original);project['images']=[];records=[]
for t,item in zip(c.PERIODS,original['images']):
 shutil.copy2(a.previous/'data/upper'/f'{t}_RGB.png',d/'upper'/f'{t}_RGB.png');shutil.copy2(a.previous/'data/original'/f'{t}_classes.png',d/'original'/f'{t}_classes.png');shutil.copy2(a.previous/'data/revised'/f'{t}_classes.png',d/'v14_reference'/f'{t}_classes.png')
 gt=np.array(Image.open(d/'original'/f'{t}_classes.png'));gate=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0;typed=(gt>0)&(gt<255);out=np.zeros_like(gt);out[gate]=255
 if t=='T1':out[gate]=4
 else:
  dist,near=ndi.distance_transform_edt(~typed,return_indices=True);nearest=gt[tuple(near)];add=gate&~typed&(dist<=8);out[add]=nearest[add]
  ds=np.stack([ndi.distance_transform_edt(gt!=k) for k in [1,2,3]]);ss=np.sort(ds,axis=0);amb=add&((ss[1]-ss[0])<1);out[amb]=255;out[gate&typed]=gt[gate&typed]
 assert np.array_equal(out[gate&typed],gt[gate&typed]);Image.fromarray(out).save(d/'revised'/f'{t}_classes.png');Image.fromarray(gate.astype('uint8')*255).save(d/'revised'/f'{t}_root_mask.png');Image.fromarray((gt>0).astype('uint8')*255).save(d/'original'/f'{t}_root_mask.png');Image.fromarray((gate&(out==255)).astype('uint8')*255).save(d/'changed'/f'{t}_untyped_model_roots.png');Image.fromarray((gate!=typed).astype('uint8')*255).save(d/'changed'/f'{t}_morphology_changes.png')
 active=copy.deepcopy(item);active.update(name=f'{t}_RGB.png',image_sha256=c.sha(d/'upper'/f'{t}_RGB.png'),width=2541,height=1108,reviewed=False);shape=(item['height'],item['width']);active['layers']={k:c.pack(c.unpack(v,shape)[:1108]) for k,v in item['layers'].items()};ui=np.zeros_like(gt)
 for cls,raw in [(1,3),(2,1),(3,2),(4,4)]:ui[out==cls]=raw
 active['layers']['mask']=c.pack(gate.astype('uint8')*255);active['layers']['temporal']=c.pack(ui);project['images'].append(active)
 records.append(dict(period=t,original_known_root_pixels=int(typed.sum()),retained_typed_pixels=int((typed&gate).sum()),original_typed_semantics_changed=0,root_morphology_pixels=int(gate.sum()),width_or_tip_added_pixels=int((gate&~typed).sum()),original_root_pixels_outside_model_support=int((typed&~gate).sum()),untyped_model_root_pixels=int((out==255).sum()),class_pixels={str(k):int((out==k).sum()) for k in [0,1,2,3,4,255]}))
for row in c.rows(d/'spatial_manifest.csv'):
 x,y=int(row['x']),int(row['y']);folder=d/'patches'/row['split'];folder.mkdir(exist_ok=True)
 for t in c.PERIODS:
  for ref in ['original','revised','v14_reference']:Image.open(d/ref/f'{t}_classes.png').crop((x,y,x+256,y+256)).save(folder/f'{row["tile_id"]}_{t}_{ref}_classes.png')
  Image.open(d/'upper'/f'{t}_RGB.png').crop((x,y,x+256,y+256)).save(folder/f'{row["tile_id"]}_{t}_RGB.png')
meta=dict(version='temporal_manual_continuity_v17',position_id='0353-2-B3',warm_start_version='v14',reference_type='Original human temporal semantics, U-Net-adjusted root morphology; unknown added roots untyped',temporal_teacher_replacement=False,original_human_typed_semantics_changed=0,morphology_source_SHA256=c.sha(a.roots/'unet/best.pt'),root_support_exact_U_Net=True,semantic_expansion_max_pixels=8,ambiguous_class_boundaries_ignored=True,untraced_roots_not_assigned_a_temporal_class=True,original_progress_SHA256=c.sha(d/'original_human_progress.json'),T1='all roots baseline',later_dates='relative to immediately preceding period',model_codes={'0':'background','1':'persistent','2':'newly_observed','3':'tip_growth','4':'baseline','255':'untyped/ignore'},UI_codes={'1':'new','2':'growth','3':'persistent','4':'baseline'},independent_external_test_images=0,RGB_unchanged=True,death_included=False)
project['annotation_revision']=meta;c.js(d/'active_progress.json',project);c.js(d/'revision_provenance.json',meta);c.js(d/'revision_log.json',records)
source=r/'source_RGB/0353-2-B3';source.mkdir(parents=True)
for t in c.PERIODS:os.link(d/'upper'/f'{t}_RGB.png',source/f'{t}_RGB.png')
print('MANUAL_SEMANTICS_U_NET_MORPHOLOGY_READY',records,flush=True)
