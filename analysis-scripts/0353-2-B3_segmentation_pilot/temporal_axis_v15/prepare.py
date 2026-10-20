"""Keep every original typed pixel. Extend only its narrow RGB/model-supported width."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;d=r/'data';assert not d.exists(),'Use a new reference version';d.mkdir(parents=True)
for f in ['upper','original','revised','changed','patches']:(d/f).mkdir()
original=a.previous/'data/original_human_progress.json';assert c.sha(original)=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09';shutil.copy2(original,d/original.name);human=json.loads(original.read_text());active=copy.deepcopy(human);active['images']=[];log=[]
for t,item in zip(c.PERIODS,human['images']):
 sh=(item['height'],item['width']);rawmask=c.unpack(item['layers']['mask'],sh)[:1108];temporal=c.unpack(item['layers']['temporal'],sh)[:1108];manual=rawmask==255;rgbfile=a.roots/'data/upper'/f'{t}_RGB.png';shutil.copy2(rgbfile,d/'upper'/rgbfile.name);image=np.array(Image.open(rgbfile));gate=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0;prob=np.load(a.roots/'unet/upper'/f'{t}_probability.npy')
 # A narrow expansion can repair thickness/tip tracing. No teacher temporal map
 # is opened, and no far model-only root is assigned a human temporal class.
 distance,nearest=ndi.distance_transform_edt(~manual,return_indices=True);contrast=c.axes.root_contrast(image);added=gate&~manual&(distance<=6)&(prob>=.45)&(contrast>-.2);morph=manual|added
 orig=np.zeros(manual.shape,np.uint8)
 if t=='T1':orig[manual]=4
 else:
  orig[manual]=255
  for raw,cls in [(3,1),(1,2),(2,3)]:orig[manual&(temporal==raw)]=cls
 orig[rawmask==128]=255;revised=orig.copy();closest=orig[tuple(nearest)];revised[added]=closest[added]
 # Do not invent an annotation boundary at an ambiguous expansion near two
 # existing temporal classes. Original strokes, including their boundaries, stay.
 if t!='T1':
  distances=np.stack([ndi.distance_transform_edt(orig!=k) for k in [1,2,3]]);ss=np.sort(distances,axis=0);ambiguous=added&((ss[1]-ss[0])<1.0);revised[ambiguous]=255
 else:ambiguous=np.zeros(manual.shape,bool)
 assert np.array_equal(revised[manual],orig[manual]);assert np.array_equal(revised[~morph],orig[~morph])
 Image.fromarray(orig).save(d/'original'/f'{t}_classes.png');Image.fromarray(manual.astype('uint8')*255).save(d/'original'/f'{t}_root_mask.png');Image.fromarray(revised).save(d/'revised'/f'{t}_classes.png');Image.fromarray(morph.astype('uint8')*255).save(d/'revised'/f'{t}_root_mask.png');Image.fromarray(added.astype('uint8')*255).save(d/'changed'/f'{t}_morphology_additions.png');Image.fromarray(ambiguous.astype('uint8')*255).save(d/'changed'/f'{t}_untyped_added_boundary.png');Image.fromarray((gate&~morph).astype('uint8')*255).save(d/'changed'/f'{t}_model_only_untraced.png')
 new=copy.deepcopy(item);new.update(name=f'{t}_RGB.png',width=2541,height=1108,image_sha256=c.sha(rgbfile),reviewed=False);new['layers']={k:c.pack(c.unpack(v,sh)[:1108]) for k,v in item['layers'].items()};ui=np.zeros_like(revised)
 for cls,raw in [(1,3),(2,1),(3,2),(4,4)]:ui[revised==cls]=raw
 new['layers']['mask']=c.pack(np.where(morph,255,rawmask).astype('uint8'));new['layers']['temporal']=c.pack(ui);active['images'].append(new);log.append(dict(period=t,original_typed_pixels_preserved=int(((orig>0)&(orig<255)&manual).sum()),original_root_pixels=int(manual.sum()),added_morphology_pixels=int(added.sum()),original_temporal_pixels_reclassified=0,ambiguous_added_untyped_pixels=int(ambiguous.sum()),RGB_SHA256=c.sha(rgbfile)))
for name in ['paired_manifest.csv','spatial_manifest.csv']:shutil.copy2(a.previous/'data'/name,d/name)
for row in c.rows(d/'spatial_manifest.csv'):
 x,y=int(row['x']),int(row['y']);out=d/'patches'/row['split'];out.mkdir(exist_ok=True)
 for t in c.PERIODS:
  for ref in ['original','revised']:Image.open(d/ref/f'{t}_classes.png').crop((x,y,x+256,y+256)).save(out/(row['tile_id']+'_'+t+'_'+ref+'_classes.png'))
  Image.open(d/'upper'/f'{t}_RGB.png').crop((x,y,x+256,y+256)).save(out/(row['tile_id']+'_'+t+'_RGB.png'))
meta=dict(version='temporal_axis_v15',position_id='0353-2-B3',original_progress_SHA256=c.sha(original),reference_type='Original human temporal semantics with narrow U-Net morphology expansion',teacher_temporal_predictions_used=False,human_original_typed_pixels_reclassified=0,all_original_root_pixels_retained=True,morphology_max_expansion_native_pixels=6,untraced_model_only_roots_not_promoted_to_human_temporal_truth=True,T1='all baseline',later_dates='relative to immediately preceding period',death_included=False,model_codes={'0':'background','1':'persistent','2':'newly_observed','3':'tip_growth','4':'baseline','255':'untyped/ignore'},UI_codes={'1':'new','2':'growth','3':'persistent','4':'baseline'},independent_external_test_images=0,historical_same_field_exposure=True,original_RGB_unchanged=True)
active['annotation_revision']=meta;c.js(d/'active_progress.json',active);c.js(d/'revision_provenance.json',meta);c.js(d/'revision_log.json',log)
c.js(d/'dataset_summary.json',dict(RGB_size_wh=[2541,1108],train_validation_internal_test_RGBs=[200,20,24],paired_examples=[150,15,18],base_train_validation_test_locations=[25,5,6],shifted_training_views=25,all_four_dates_same_split=True,independent_external_test_images=0))
print('HUMAN_FIRST_REFERENCE_READY',log,flush=True)
