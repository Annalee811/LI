"""Preserve originals; make model-derived temporal reference and paired splits."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run=a.run;data=run/'data';data.mkdir(exist_ok=True)
orig=a.previous/'data/original_human_progress.json';assert c.sha(orig)=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09'
shutil.copy2(orig,data/'original_human_progress.json');shutil.copy2(a.previous/'data/active_progress.json',data/'previous_v13_progress.json');human=json.loads(orig.read_text());revision=copy.deepcopy(human);revision['images']=[];log=[];counts=[]
for folder in ['upper','original','revised','changed','patches']:(data/folder).mkdir(exist_ok=True)
for t,item in zip(c.PERIODS,human['images']):
 shape=(item['height'],item['width']);mask=c.unpack(item['layers']['mask'],shape)[:1108];temporal=c.unpack(item['layers']['temporal'],shape)[:1108];target=c.original_target(mask,temporal,t=='T1');pred=np.array(Image.open(run/'teacher_unet'/f'{t}_temporal_classes.png'));gate=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0
 assert pred.shape==target.shape==(1108,2541) and np.array_equal(pred>0,gate)
 if t=='T1':assert set(np.unique(pred))<={0,4}
 else:assert set(np.unique(pred))<={0,1,2,3}
 rgbfile=a.roots/'data/upper'/f'{t}_RGB.png';shutil.copy2(rgbfile,data/'upper'/rgbfile.name);assert np.array_equal(np.array(Image.open(rgbfile)),np.array(Image.open(a.previous/'data/upper'/f'{t}_RGB.png')))
 Image.fromarray(target).save(data/'original'/f'{t}_classes.png');Image.fromarray(mask).save(data/'original'/f'{t}_root_mask.png');Image.fromarray(pred).save(data/'revised'/f'{t}_classes.png');Image.fromarray(gate.astype('uint8')*255).save(data/'revised'/f'{t}_root_mask.png')
 if t=='T1':conf=np.load(a.roots/'unet/upper'/f'{t}_probability.npy');conf=np.where(gate,conf,0);Image.fromarray(np.rint(conf*65535).astype('uint16')).save(data/'revised'/f'{t}_confidence_uint16.png')
 else:shutil.copy2(run/'teacher_unet'/f'{t}_class_confidence_uint16.png',data/'revised'/f'{t}_confidence_uint16.png');conf=np.array(Image.open(data/'revised'/f'{t}_confidence_uint16.png')).astype(float)/65535
 raw=np.zeros_like(pred);mapping={0:0,1:3,2:1,3:2,4:4}
 for old,new in mapping.items():raw[pred==old]=new
 new=copy.deepcopy(item);new.update(name=f'{t}_RGB.png',image_sha256=c.sha(rgbfile),width=2541,height=1108,reviewed=False,evidence='U-Net-derived revised reference; inspect against adjacent RGB; not independent manual ground truth')
 new['layers']={k:c.pack(c.unpack(v,shape)[:1108]) for k,v in item['layers'].items()};new['layers']['mask']=c.pack(gate.astype('uint8')*255);new['layers']['temporal']=c.pack(raw);revision['images'].append(new)
 change=pred!=target;Image.fromarray(change.astype('uint8')*255).save(data/'changed'/f'{t}_changed_pixels.png');review=gate&((conf<.5)|(mask==128));Image.fromarray(review.astype('uint8')*255).save(data/'changed'/f'{t}_review_pixels.png')
 matrix=np.zeros((5,5),np.int64);valid=target!=255;np.add.at(matrix,(target[valid],pred[valid]),1)
 log.append(dict(period=t,changed_pixels=int(change.sum()),added_root_pixels=int((gate&(mask!=255)).sum()),removed_original_root_pixels=int(((mask==255)&~gate).sum()),review_pixels=int(review.sum()),original_to_revised_class_counts=matrix.tolist(),original_full_RGB_SHA256=item['image_sha256'],upper_RGB_SHA256=c.sha(rgbfile)))
 for k in [0,1,2,3,4]:counts.append(dict(period=t,class_id=k,class_name={0:'background',1:'persistent',2:'newly_observed',3:'tip_growth',4:'T1_baseline'}[k],pixels=int((pred==k).sum())))
provenance=dict(version='temporal_reference_v14',position_id='0353-2-B3',scope='Four 2541x1108 upper RGBs only; lower source images and original progress untouched',reference_type='Model-derived U-Net temporal revised reference, not independent human truth',teacher_root_checkpoint_SHA256=c.sha(a.roots/'unet/best.pt'),teacher_temporal_head_SHA256=c.sha(a.previous/'unet/final_head.pt'),teacher_temporal_head_was_all_upper_fit=True,teacher_inference_sha256=c.sha(run/'teacher_unet/inference.json'),original_progress_SHA256=c.sha(orig),T1='all teacher roots baseline',later_dates='relative to immediately preceding date',death_included=False,root_and_temporal_masks_replaced_in_upper_reference=True,model_weights_modified=False,training_uses_revised_references=True,checkpoint_selection='Revised validation agreement; original labels retained as audit',evaluation_warning='Teacher saw same-field all-upper labels; v13 warm starts used a different same-field split. Neither comparison is a new blind or external independent test.',raw_UI_temporal_codes={'0':'unclassified','1':'newly_observed','2':'tip_growth','3':'persistent','4':'T1_baseline'},model_class_codes={'0':'background','1':'persistent','2':'newly_observed','3':'tip_growth','4':'T1_baseline'})
revision['annotation_revision']=provenance;c.js(data/'active_progress.json',revision);c.js(data/'revision_provenance.json',provenance);c.js(data/'revision_log.json',log);c.csvout(data/'revised_class_counts.csv',counts)
source=c.rows(a.roots/'data/manifest.csv');groups={}
for r in source:
 key=r['tile_id'];core={k:r[k] for k in ['tile_id','x','y','width','height','split']}
 if key in groups:assert groups[key]==core
 else:groups[key]=core
pairs=[]
for r in groups.values():
 x,y=int(r['x']),int(r['y']);assert int(r['width'])==int(r['height'])==256
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  rec=dict(pair_id=r['tile_id']+'_'+prev+'_'+t,previous=prev,current=t,**r);pairs.append(rec)
  dest=data/'patches'/r['split'];dest.mkdir(exist_ok=True)
  for ref in ['original','revised']:
   yy=np.array(Image.open(data/ref/f'{t}_classes.png'))[y:y+256,x:x+256];Image.fromarray(yy).save(dest/(rec['pair_id']+'_'+ref+'_target.png'))
  # RGB windows are stored once per date/tile, not duplicated in every pair.
  for tt in [prev,t]:
   f=dest/(r['tile_id']+'_'+tt+'_RGB.png')
   if not f.exists():Image.open(data/'upper'/f'{tt}_RGB.png').crop((x,y,x+256,y+256)).save(f)
counts={s:sum(r['split']==s for r in pairs) for s in ['train','val','test']};assert counts==dict(train=150,val=15,test=18)
for x in groups.values():
 for y in groups.values():
  if x['split']==y['split']:continue
  ax,ay,bx,by=map(int,[x['x'],x['y'],y['x'],y['y']]);assert max(ax,bx)>=min(ax+256,bx+256) or max(ay,by)>=min(ay+256,by+256),'Cross-split overlap'
c.csvout(data/'paired_manifest.csv',pairs);c.csvout(data/'spatial_manifest.csv',list(groups.values()));c.js(data/'dataset_summary.json',dict(original_RGBs=4,source_size_wh=[2541,1108],spatial_windows={s:sum(r['split']==s for r in groups.values()) for s in counts},paired_examples=counts,train_base_spatial_positions=25,train_shifted_positions=25,validation_spatial_positions=5,test_spatial_positions=6,dates_always_same_split=True,cross_split_pixel_overlap=0,temporal_pairs=['T1->T2','T2->T3','T3->T4'],T1_is_deterministic_baseline_not_a_temporal_training_pair=True,independent_external_test_images=0,historical_warmstart_exposure_possible=True))
assert c.sha(orig)==provenance['original_progress_SHA256'];print('PREPARED_REVISED_TEMPORAL_REFERENCE',counts,flush=True)
