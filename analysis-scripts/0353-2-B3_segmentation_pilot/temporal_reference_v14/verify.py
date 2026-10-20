"""Check source preservation, spatial partitioning, trained heads and real outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();r=a.run;checks={};torch.set_num_threads(3)
assert c.sha(r/'data/original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09';checks['original_human_JSON_byte_preserved']=True
project=json.loads((r/'data/active_progress.json').read_text());assert project['annotation_revision']['reference_type'].startswith('Model-derived');checks['source_provenance_preserved']=True
for t,it in zip(c.PERIODS,project['images']):
 assert it['name']==t+'_RGB.png' and it['width']==2541 and it['height']==1108
 assert c.sha(r/'data/upper'/it['name'])==it['image_sha256']==c.sha(a.roots/'data/upper'/it['name'])
 raw=c.unpack(it['layers']['temporal'],(1108,2541));mask=c.unpack(it['layers']['mask'],(1108,2541));model=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'));mapping=np.array([0,2,3,1,4],np.uint8);assert np.array_equal(mapping[raw],model)
 assert np.array_equal(model>0,mask==255)
 assert set(np.unique(model))<=({0,4} if t=='T1' else {0,1,2,3})
checks['four_RGB_and_UI_class_mapping']=True
windows=c.rows(r/'data/spatial_manifest.csv');pairs=c.rows(r/'data/paired_manifest.csv');assert {s:sum(x['split']==s for x in pairs) for s in ['train','val','test']}==dict(train=150,val=15,test=18)
for x in windows:
 pp=[v for v in pairs if v['tile_id']==x['tile_id']];assert len(pp)==3 and len({v['split'] for v in pp})==1
 for y in windows:
  if x['split']==y['split']:continue
  ax,ay,bx,by=map(int,[x['x'],x['y'],y['x'],y['y']]);assert max(ax,bx)>=min(ax+256,bx+256) or max(ay,by)>=min(ay+256,by+256)
checks['all_dates_grouped_zero_cross_split_pixels']=True
base=None;details={};device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
for m in c.MODELS:
 f=r/m;meta=json.loads((f/'training_run.json').read_text());assert meta['epochs_completed']==24 and meta['optimizer_updates']==384 and meta['unique_training_pairs']==150
 draws=c.rows(f/'training_draws.csv');assert len(draws)==1536
 if base is None:base=draws
 else:assert draws==base
 root=a.roots/m/('best_weights_fp16.npz' if m=='deeplab' else 'best.pt');assert c.sha(root)==meta['root_checkpoint_SHA256'];assert c.sha(f/'best_temporal_head.pt')==meta['selected_head_SHA256']
 infer=json.loads((f/'four_period_prediction/inference.json').read_text());assert infer['head_sha256']==meta['selected_head_SHA256'] and infer['manual_masks_used'] is False
 for t in c.PERIODS:
  cl=np.array(Image.open(f/'four_period_prediction'/f'{t}_temporal_classes.png'));gate=np.array(Image.open(a.roots/m/'upper'/f'{t}_root_mask.png'))>0;assert np.array_equal(cl>0,gate)
  assert set(np.unique(cl))<=({0,4} if t=='T1' else {0,1,2,3})
  im=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'));assert np.array_equal(im,np.array(Image.open(f/'four_period_prediction'/f'{t}_RGB.png')))
  rgba=np.array(Image.open(f/'four_period_prediction'/f'{t}_original_roots_RGBA.png'));assert np.array_equal(rgba[gate,:3],im[gate])
 data=c.Data(r,a.roots,m);head=c.load_head(f/'best_temporal_head.pt',device);report=c.evaluate(head,data,'test',device,amp=amp);saved=json.loads((f/'metrics.json').read_text())['test'];assert report==saved,'Checkpoint reload metrics differ'
 select=json.loads((f/'decoder_selection.json').read_text());assert not select['test_used_for_selection'];best=max(x['revised_macro_Dice'] for x in select['candidates']);assert abs(best-select['selected_revised_validation_macro_Dice'])<1e-10
 details[m]=dict(epochs=24,optimizer_updates=384,selected_epoch=meta['selected_epoch'],selected_head_SHA256=meta['selected_head_SHA256'],frozen_root_SHA256=c.sha(root),all_test_metrics_reproduced_by_checkpoint_reload=True,revised_test_macro_Dice=saved['revised']['end_to_end']['macro_Dice'],original_test_macro_Dice=saved['original']['end_to_end']['macro_Dice'])
checks['identical_three_model_draws']=True;checks['actual_trained_heads_reload_all_18_test_pairs']=True;checks['root_weights_RGB_and_foreground_pixels_unchanged']=True;checks['full_outputs_match_calibrated_head_hash']=True
editor=json.loads((r/'editor_verification.json').read_text());assert not editor['page_errors'] and all(editor[k] for k in ['preloaded_four_RGB','baseline_only_in_T1','real_pointer_brush_and_CtrlZ','zoom_preserved_across_dates','saved_JSON_retains_source_and_four_images','saved_JSON_reimport']);checks['browser_brush_undo_zoom_save_reimport']=True
c.js(r/'verification.json',dict(status='PASS',checks=checks,models=details,limitations='Same-field model-derived reference agreement; no independent external test or manual physical length truth'))
print('VERIFIED_ALL_THREE_TEMPORAL_MODELS',flush=True)
