"""Create a versioned U-Net-derived reference, keeping the human project intact."""
import argparse,copy,json,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;d=r/'data';assert not d.exists();d.mkdir(parents=True)
for folder in ['upper','original','revised','changed','patches']:(d/folder).mkdir()
for name in ['original_human_progress.json','paired_manifest.csv','spatial_manifest.csv','dataset_summary.json']:shutil.copy2(a.previous/'data'/name,d/name)
shutil.copy2(a.previous/'data/active_progress.json',d/'previous_v15_human_first_progress.json')
active=json.loads((a.previous/'data/active_progress.json').read_text());log=[]
for t,item in zip(c.PERIODS,active['images']):
 shutil.copy2(a.previous/'data/upper'/f'{t}_RGB.png',d/'upper'/f'{t}_RGB.png')
 for suffix in ['classes','root_mask']:shutil.copy2(a.previous/'data/original'/f'{t}_{suffix}.png',d/'original'/f'{t}_{suffix}.png')
 teacher=a.previous/'unet/four_period_prediction'/f'{t}_temporal_classes.png';gt=np.array(Image.open(teacher));orig=np.array(Image.open(d/'original'/f'{t}_classes.png'));assert gt.shape==(1108,2541);assert set(np.unique(gt))<=({0,4} if t=='T1' else {0,1,2,3})
 Image.fromarray(gt).save(d/'revised'/f'{t}_classes.png');Image.fromarray((gt>0).astype('uint8')*255).save(d/'revised'/f'{t}_root_mask.png');changed=(gt!=orig)&(orig!=255);Image.fromarray(changed.astype('uint8')*255).save(d/'changed'/f'{t}_changed.png')
 ui=np.zeros_like(gt)
 for k,v in [(1,3),(2,1),(3,2),(4,4)]:ui[gt==k]=v
 item['layers']['mask']=c.pack((gt>0).astype('uint8')*255);item['layers']['temporal']=c.pack(ui);item['reviewed']=False
 log.append(dict(period=t,source=str(teacher),source_SHA256=c.sha(teacher),original_human_typed_root_pixels=int(((orig>0)&(orig<255)).sum()),revised_root_pixels=int((gt>0).sum()),changed_known_pixels=int(changed.sum()),human_root_temporal_reclassified_pixels=int(((orig>0)&(orig<255)&(gt>0)&(gt!=orig)).sum()),class_pixels={str(k):int((gt==k).sum()) for k in range(5)},RGB_SHA256=c.sha(d/'upper'/f'{t}_RGB.png')))
for row in c.rows(d/'spatial_manifest.csv'):
 x,y=int(row['x']),int(row['y']);out=d/'patches'/row['split'];out.mkdir(exist_ok=True)
 for t in c.PERIODS:
  for ref in ['original','revised']:Image.open(d/ref/f'{t}_classes.png').crop((x,y,x+256,y+256)).save(out/f'{row["tile_id"]}_{t}_{ref}_classes.png')
  Image.open(d/'upper'/f'{t}_RGB.png').crop((x,y,x+256,y+256)).save(out/f'{row["tile_id"]}_{t}_RGB.png')
for folder in ['feature_cache','axis_cache']:
 if (a.previous/folder).is_dir():os.symlink((a.previous/folder).resolve(),r/folder,target_is_directory=True)
 else:(r/folder).mkdir()
meta=dict(version='temporal_reference_v16',position_id='0353-2-B3',reference_type='U-Net-derived temporal and morphology reference; pending user review',teacher_head_SHA256=c.sha(a.previous/'unet/best_temporal_head.pt'),original_human_progress_SHA256=c.sha(d/'original_human_progress.json'),original_human_preserved_separately=True,teacher_uses_same_field_historical_training=True,reference_is_independent_human_truth=False,T1='baseline only',T2_T3_T4='Relative to the immediately preceding date',death_included=False,model_codes={'0':'background','1':'persistent','2':'new','3':'tip_growth','4':'baseline','255':'ignore'},UI_codes={'1':'new','2':'growth','3':'persistent','4':'baseline'},independent_external_test_images=0,RGB_native_unchanged=True,teacher_confidence_filter=False,all_reference_root_pixels_included=True)
active['annotation_revision']=meta;c.js(d/'active_progress.json',active);c.js(d/'revision_provenance.json',meta);c.js(d/'revision_log.json',log)
source=r/'source_RGB/0353-2-B3';source.mkdir(parents=True)
for t in c.PERIODS:os.link(d/'upper'/f'{t}_RGB.png',source/f'{t}_RGB.png')
print('REFERENCE_READY',log,flush=True)
