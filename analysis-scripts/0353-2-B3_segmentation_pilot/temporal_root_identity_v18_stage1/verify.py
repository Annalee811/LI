"""Audit manual semantics, native RGB, same splits and actual label-free reloads."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();r=a.run;checks={};assert c.sha(r/'data/original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09';checks['original_human_JSON_byte_identical']=True;active=json.loads((r/'data/active_progress.json').read_text());audit=[]
log=json.loads((r/'data/revision_log.json').read_text());summaries=json.loads((r/'data/revision_summary.json').read_text())
for t,item in zip(c.PERIODS,active['images']):
 gt=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'));frozen=np.array(Image.open(r/'data/frozen'/f'{t}_classes.png'));orig=np.array(Image.open(r/'data/original'/f'{t}_classes.png'));gate=np.array(Image.open(a.roots/'unet/upper'/f'{t}_root_mask.png'))>0;assert np.array_equal(gt>0,gate) and np.array_equal(frozen>0,gate);assert np.array_equal(gt==255,frozen==255)
 assert np.array_equal(frozen,np.array(Image.open(a.previous/'data/revised'/f'{t}_classes.png')));changed=gt!=frozen;count=int(changed.sum());assert count==next(x['semantic_pixels_changed'] for x in summaries if x['period']==t);assert count==sum(x['changed_pixels'] for x in log if x['period']==t)
 if t=='T1':assert count==0 and set(np.unique(gt))<={0,4}
 for row in [x for x in log if x['period']==t]:
  assert row['current_binary_U_Net_score']>=.8 and row['human_reviewed'] is False
  if row['reason'].startswith('Persistent'):assert row['previous_U_Net_visibility']>=.8 and row['joint_previous_visibility']>=.7
  else:assert row['previous_U_Net_visibility']<=.1 and row['previous_manual_absence']>=.9 and sum(row['U_Net_temporal_scores_P_N_G'][1:])>=.8
 raw=c.unpack(item['layers']['temporal'],(1108,2541));mask=c.unpack(item['layers']['mask'],(1108,2541));recovered=np.zeros_like(gt)
 for ui,code in [(3,1),(1,2),(2,3),(4,4)]:recovered[raw==ui]=code
 recovered[(mask==255)&(raw==0)]=255;assert np.array_equal(gt,recovered);assert np.array_equal(mask==255,gate);assert c.sha(r/'data/upper'/f'{t}_RGB.png')==c.sha(a.roots/'data/upper'/f'{t}_RGB.png')==item['image_sha256'];audit.append(dict(period=t,semantic_pixels_changed=count,unknown_pixels_unchanged=True,root_geometry_unchanged=True))
checks.update(selective_semantic_changes_all_logged=True,changes_meet_joint_U_Net_manual_previous_period_rules=True,frozen_V17_evaluation_reference_byte_identical=True,root_geometry_and_RGB_preserved=True)
pairs=c.rows(r/'data/paired_manifest.csv');spatial=c.rows(r/'data/spatial_manifest.csv');assert [sum(q['split']==s for q in pairs) for s in ['train','val','test']]==[150,15,18]
for x in spatial:
 assert len({p['split'] for p in pairs if p['tile_id']==x['tile_id']})==1
 for y in spatial:
  if x['split']==y['split']:continue
  ax,ay,bx,by=map(int,[x['x'],x['y'],y['x'],y['y']]);assert max(ax,bx)>=min(ax+256,bx+256) or max(ay,by)>=min(ay+256,by+256)
checks['dates_grouped_and_no_cross_split_pixel_overlap']=True;draws=None;models={}
for m in c.MODELS:
 folder=r/m;meta=json.loads((folder/'training_run.json').read_text());assert meta['epochs_completed']==32 and meta['optimizer_updates']==512 and meta['unique_training_pairs']==150;actual=c.rows(folder/'training_draws.csv');assert len(actual)==2048
 if draws is None:draws=actual
 else:assert draws==actual
 root=a.roots/m/('best_weights_fp16.npz' if m=='deeplab' else 'best.pt');assert c.sha(root)==meta['root_SHA256'];assert c.sha(folder/'best_temporal_head.pt')==meta['head_SHA256'];assert c.sha(r/'warm_starts'/m/'best_temporal_head.pt')==meta['warm_start_SHA256'];native=r/'label_free_reload'/m;reload=json.loads((native/'inference.json').read_text());assert reload['manual_labels_read'] is False and reload['verification_against_existing_output'] is True and reload['head_SHA256']==meta['head_SHA256'];router=json.loads((r/'routing'/m/'router.json').read_text());assert router['samples']>=8 and router['positives']>0 and router['negatives']>0 and router['independent_root_identity_labels'] is False
 selection=json.loads((folder/'decoder_selection.json').read_text());assert selection['test_used_for_selection'] is False and len(selection['candidates'])==36;assert abs(max(v['validation_frozen_macro_Dice'] for v in selection['candidates'])-selection['validation_frozen_macro_Dice'])<1e-10
 maps={};reviewcounts=[]
 for t in c.PERIODS:
  classes=np.array(Image.open(folder/'four_period_prediction'/f'{t}_temporal_classes.png'));mask=np.array(Image.open(a.roots/m/'upper'/f'{t}_root_mask.png'))>0;assert np.array_equal(classes>0,mask);assert np.array_equal(classes,np.array(Image.open(native/f'{t}_temporal_classes.png')));rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'));assert np.array_equal(rgb,np.array(Image.open(folder/'four_period_prediction'/f'{t}_RGB.png')));rgba=np.array(Image.open(folder/'four_period_prediction'/f'{t}_original_roots_RGBA.png'));assert np.array_equal(rgb[mask],rgba[mask,:3]);maps[t]=classes
  if t=='T1':assert set(np.unique(classes))<={0,4};continue
  axis=c.load_axes(r/'axis_cache'/m/f'{t}.npz');records=json.loads((folder/'four_period_prediction'/f'{t}_root_axes.json').read_text());review=axis['uncertain'].copy()&mask;assert len(records)==len(axis['paths'])
  for row in records:
   region=axis['axis_map']==row['axis_id'];values=np.unique(classes[region]);assert not(2 in values and (1 in values or 3 in values))
   if row['unresolved_crossing'] or row['mode']=='previous_tip_growth_continuation' or row['decision_margin'] is not None and row['decision_margin']<.15:review|=region
  Image.fromarray(review.astype('uint8')*255).save(folder/'four_period_prediction'/f'{t}_temporal_review_mask.png');reviewcounts.append(dict(period=t,review_root_pixels=int(review.sum())))
 metrics=json.loads((folder/'metrics.json').read_text())
 for split in ['val','test']:
  for ref in ['original','revised','frozen']:
   cm=np.zeros((4,4),np.int64)
   for row in pairs:
    if row['split']!=split:continue
    x,y=int(row['x']),int(row['y']);t=row['current'];gt=np.array(Image.open(r/'data'/ref/f'{t}_classes.png'))[y:y+256,x:x+256];pred=maps[t][y:y+256,x:x+256];cm+=c.old.confusion(gt,pred);assert np.array_equal(pred,np.array(Image.open(folder/'evaluation'/split/(row['pair_id']+'_prediction.png'))))
   assert c.old.metrics(cm)==metrics[split][ref]['end_to_end']
 models[m]=dict(epochs=32,selected_epoch=meta['selected_epoch'],head_SHA256=meta['head_SHA256'],warm_head_SHA256=meta['warm_start_SHA256'],root_SHA256=meta['root_SHA256'],selected_decoder=selection['decoder'],new_head_blend_weight=selection['new_head_weight'],reference_test_macro_Dice=metrics['test']['frozen']['end_to_end']['macro_Dice'],original_audit_test_macro_Dice=metrics['test']['original']['end_to_end']['macro_Dice'],review_counts=reviewcounts,label_free_reload_all_dates_exact=True)
checks.update(identical_training_draws_three_models=True,all_root_weights_frozen=True,all_heads_actually_trained32_epochs=True,all_test_confusions_recomputed=True,label_free_reload_all_four_dates_exact=True,no_NEW_GROWTH_or_NEW_PERSISTENT_mixture_in_any_accepted_axis=True)
editor=json.loads((r/'editor_verification.json').read_text());assert not editor['page_errors'];checks['real_browser_editor_checked']=True;c.js(r/'verification.json',dict(status='PASS',checks=checks,morphology_audit=audit,models=models,biological_root_instance_accuracy_not_independently_measured=True,independent_external_test_images=0,whole_field_context_crosses_internal_windows=True,historical_same_field_exposure=True));print('ROOT_IDENTITY_AND_SELECTIVE_CORRECTIONS_VERIFIED',flush=True)
