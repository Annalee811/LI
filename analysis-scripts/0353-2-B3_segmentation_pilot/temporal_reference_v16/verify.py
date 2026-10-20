"""Audit versioned teacher reference, whole-axis grammar and label-free model reload."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();r=a.run;checks={};assert c.sha(r/'data/original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09';checks['original_JSON_byte_preserved']=True;original=json.loads((r/'data/original_human_progress.json').read_text());active=json.loads((r/'data/active_progress.json').read_text());revision=[]
for t,item,edited in zip(c.PERIODS,original['images'],active['images']):
 gt=np.array(Image.open(r/'data/revised'/f'{t}_classes.png'));raw=c.unpack(edited['layers']['temporal'],(1108,2541));mask=c.unpack(edited['layers']['mask'],(1108,2541));recovered=np.zeros_like(raw)
 for ui,cls in [(3,1),(1,2),(2,3),(4,4)]:recovered[raw==ui]=cls
 assert np.array_equal(recovered,gt) and np.array_equal(mask==255,gt>0)
 teacher=r.parent/'B3_temporal_axis_v15/unet/four_period_prediction'/f'{t}_temporal_classes.png';assert np.array_equal(gt,np.array(Image.open(teacher)))
 if t=='T1':assert set(np.unique(gt))<={0,4}
 else:assert set(np.unique(gt))<={0,1,2,3}
 assert c.sha(r/'data/upper'/f'{t}_RGB.png')==c.sha(a.roots/'data/upper'/f'{t}_RGB.png')==edited['image_sha256']
 revision.append(dict(period=t,teacher_reference_exact=True,root_pixels=int((gt>0).sum())))
checks['U_Net_teacher_reference_exact_and_separately_versioned']=True;checks['native_RGB_byte_preserved']=True
windows=c.rows(r/'data/spatial_manifest.csv');pairs=c.rows(r/'data/paired_manifest.csv');assert {s:sum(v['split']==s for v in pairs) for s in ['train','val','test']}==dict(train=150,val=15,test=18)
for x in windows:
 assert len({v['split'] for v in pairs if v['tile_id']==x['tile_id']})==1
 for y in windows:
  if x['split']==y['split']:continue
  ax,ay,bx,by=map(int,[x['x'],x['y'],y['x'],y['y']]);assert max(ax,bx)>=min(ax+256,bx+256) or max(ay,by)>=min(ay+256,by+256)
checks['all_dates_grouped_and_cross_split_pixel_overlap_zero']=True;draws=None;continuity=[];overview=[];models={}
for m in c.MODELS:
 folder=r/m;meta=json.loads((folder/'training_run.json').read_text());assert meta['epochs_completed']==24 and meta['optimizer_updates']==384 and meta['unique_training_pairs']==150;actual=c.rows(folder/'training_draws.csv');assert len(actual)==1536
 if draws is None:draws=actual
 else:assert actual==draws
 weight=a.roots/m/('best_weights_fp16.npz' if m=='deeplab' else 'best.pt');assert c.sha(weight)==meta['root_SHA256'];head=folder/'best_temporal_head.pt';assert c.sha(head)==meta['head_SHA256'];native=r/'label_free_reload'/m;reload=json.loads((native/'inference.json').read_text());assert reload['manual_labels_read'] is False and reload['verification_against_existing_output'] is True and reload['head_SHA256']==c.sha(head)
 selected=json.loads((folder/'decoder_selection.json').read_text());assert selected['test_used_for_selection'] is False;assert abs(max(v['validation_macro_Dice'] for v in selected['candidates'])-next(v['validation_macro_Dice'] for v in selected['candidates'] if v['strength']==selected['selected_strength']))<1e-10
 maps={}
 for t in c.PERIODS:
  cl=np.array(Image.open(folder/'four_period_prediction'/f'{t}_temporal_classes.png'));gate=np.array(Image.open(a.roots/m/'upper'/f'{t}_root_mask.png'))>0;assert np.array_equal(cl>0,gate);assert np.array_equal(cl,np.array(Image.open(native/f'{t}_temporal_classes.png')));rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'));assert np.array_equal(rgb,np.array(Image.open(folder/'four_period_prediction'/f'{t}_RGB.png')));rgba=np.array(Image.open(folder/'four_period_prediction'/f'{t}_original_roots_RGBA.png'));assert np.array_equal(rgb[gate],rgba[gate,:3]);maps[t]=cl
  if t=='T1':assert set(np.unique(cl))<={0,4};continue
  axis=c.load_axes(r/'axis_cache'/m/(t+'.npz'));rr=json.loads((folder/'four_period_prediction'/(t+'_root_axes.json')).read_text());assert len(rr)==len(axis['paths']);rawprob=np.load(folder/'four_period_prediction'/(t+'_class_probability.npz'))['probability'];rawclass=np.where(gate,rawprob.argmax(0)+1,0);review=axis['uncertain'].copy()&gate;badbefore=badafter=0
  for row in rr:
   region=axis['axis_map']==row['axis_id'];values=np.unique(cl[region]);rv=np.unique(rawclass[region]);badafter+=int(2 in values and 3 in values);badbefore+=int(2 in rv and 3 in rv);assert not(row['new_nodes'] and row['growth_nodes'])
   if row['mode']=='tip_growth_continuation_at_crossing':assert row['growth_continuation_previous_axis']>0
   if row['unresolved_crossing'] and row['decision_margin'] is not None and abs(row['decision_margin'])<.20:review|=region
  assert badafter==0;review|=((rawclass==3)&(cl==2))|((rawclass==2)&(cl==3));Image.fromarray(review.astype('uint8')*255).save(folder/'four_period_prediction'/(t+'_temporal_review_mask.png'));overlay=rgb.copy();overlay[review]=np.rint(.35*rgb[review]+.65*np.array([160,160,160])).astype('uint8');Image.fromarray(overlay).save(folder/'four_period_prediction'/(t+'_temporal_review_overlay.png'));continuity.append(dict(model=m,period=t,traced_axes=len(rr),new_growth_mixed_axes_before=badbefore,new_growth_mixed_axes_after=badafter,unresolved_crossing_axes=sum(z['unresolved_crossing'] for z in rr),temporal_review_pixels=int(review.sum()),growth_continuation_fragments=sum(z['mode']=='tip_growth_continuation_at_crossing' for z in rr)))
 metric=json.loads((folder/'metrics.json').read_text());rawmetric=json.loads((folder/'raw_pixel_test_metrics.json').read_text())
 for split in ['val','test']:
  for ref in ['original','revised']:
   cm=np.zeros((4,4),np.int64)
   for row in [v for v in pairs if v['split']==split]:
    x,y=int(row['x']),int(row['y']);t=row['current'];gt=np.array(Image.open(r/'data'/ref/f'{t}_classes.png'))[y:y+256,x:x+256];prediction=maps[t][y:y+256,x:x+256];cm+=c.old.confusion(gt,prediction);assert np.array_equal(prediction,np.array(Image.open(folder/'evaluation'/split/(row['pair_id']+'_prediction.png'))))
   assert c.old.metrics(cm)==metric[split][ref]['end_to_end']
 for ref in ['original','revised']:
  z=metric['test'][ref]['end_to_end'];overview.append(dict(model=m,reference=ref,macro_Dice=z['macro_Dice'],macro_Precision=float(np.mean([q['Precision'] or 0 for q in z['classes']])),macro_Recall=float(np.mean([q['Recall'] or 0 for q in z['classes']])),macro_IoU=float(np.mean([q['IoU'] or 0 for q in z['classes']])),raw_pixel_macro_Dice=rawmetric[ref]['end_to_end']['macro_Dice']))
 models[m]=dict(selected_epoch=meta['selected_epoch'],epochs=24,optimizer_updates=384,head_SHA256=c.sha(head),root_SHA256=c.sha(weight),label_free_reload_all_four_periods_identical=True,original_test_macro_Dice=metric['test']['original']['end_to_end']['macro_Dice'],human_morphology_reference_test_macro_Dice=metric['test']['revised']['end_to_end']['macro_Dice'])
checks.update(identical_draws_three_models=True,root_weights_frozen=True,label_free_actual_head_reload_all_four_frames=True,all_val_test_confusions_recomputed=True,no_new_growth_mixture_inside_any_traced_axis=True)
editor=json.loads((r/'editor_verification.json').read_text());assert not editor['page_errors'] and all(editor[k] for k in ['preloaded_four_RGB','baseline_only_in_T1','real_pointer_brush_and_CtrlZ','zoom_preserved_across_dates','saved_JSON_retains_source_and_four_images','saved_JSON_reimport']);checks['browser_brush_undo_zoom_save_reimport']=True
c.csvout(r/'tables/root_axis_continuity.csv',continuity);c.csvout(r/'tables/metrics_overview.csv',overview);c.js(r/'verification.json',dict(status='PASS',checks=checks,reference_revision=revision,models=models,grammar_is_a_constraint_not_proof_of_biological_identity=True,independent_external_test_images=0,full_axis_context_crosses_internal_split_windows=True))
print('TEACHER_REFERENCE_AND_THREE_LABEL_FREE_MODELS_VERIFIED',flush=True)
