#!/usr/bin/env python3
"""Software fixtures only: these are not human root annotations or experiment scores."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,json,hashlib
import numpy as np
from PIL import Image
import torch
import pilot

checks=[]
gt=np.array([[255,255,0],[0,128,0]],np.uint8)
pred=np.array([[1,0,1],[0,1,0]],bool)
assert pilot.counts(gt,pred)==(1,1,1,2)
m=pilot.metrics_from_counts(*pilot.counts(gt,pred));assert m['Precision']==.5 and m['Recall']==.5 and m['Dice']==.5 and np.isclose(m['IoU'],1/3)
checks.append('TP_FP_FN_ignore_pixel_and_empty_denominator_contracts')
e=pilot.metrics_from_counts(0,0,0,9);assert e['Dice']==1 and e['IoU']==1 and e['Precision'] is None and e['Recall'] is None
s=np.zeros((9,9),bool);s[4,2:7]=1;assert pilot.skeleton_length(s)==4
s=np.eye(7,dtype=bool);assert np.isclose(pilot.skeleton_length(s),6*np.sqrt(2))
checks.append('horizontal_and_diagonal_skeleton_lengths_in_pixel_units')
a=pilot.agreement([5,10,20],[5,10,20]);assert a['R2_identity']==1 and a['RMSE']==0 and a['CCC']==1
a=pilot.agreement([5,10,20],[7,12,22]);assert a['R2_identity']<1 and a['RMSE']==2 and np.isclose(a['Pearson_r2'],1)
checks.append('identity_R2_distinguished_from_squared_correlation')
torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix='root_software_fixture_',dir='/tmp') as tmp:
    root=Path(tmp);kit=root/'SOFTWARE_FIXTURE';(kit/'images').mkdir(parents=True);(kit/'labels').mkdir();rows=[]
    for i,split in enumerate(['train','train','val','test']):
        rng=np.random.default_rng(i);rgb=rng.integers(20,190,(64,64,3),dtype=np.uint8);mask=np.zeros((64,64),np.uint8);mask[12:50,30:34]=255;mask[:3,:]=128;key=f'SOFTWARE_ONLY_{i}';image=kit/'images'/(key+'.png');label=kit/'labels'/(key+'.png');Image.fromarray(rgb).save(image);Image.fromarray(mask).save(label)
        record=dict(image_id=key,image_path='images/'+image.name,image_sha256=pilot.digest(image),height=64,width=64,split=split,window_id='SOFTWARE_WINDOW_'+str(i),spatial_group='SOFTWARE_GROUP_'+str(i),period=1);rows.append(record)
        pilot.save_json(label.with_suffix('.json'),dict(image_sha256=record['image_sha256'],mask_sha256=pilot.digest(label),reviewed_entire_image=True,annotator='SOFTWARE_TEST_FIXTURE_NOT_A_HUMAN_ROOT_ANNOTATION'))
    pilot.write_csv(kit/'split_manifest.csv',rows)
    pilot.write_csv(kit/'manual_lengths.csv',[dict(image_id=r['image_id'],manual_length_px='',method='',annotator='',fully_traced='',mm_per_pixel='') for r in rows])
    pilot.export_rootpainter(SimpleNamespace(kit=kit,output=root/'rootpainter_export'))
    exported=root/'rootpainter_export';assert not (exported/'dataset'/'SOFTWARE_ONLY_3.png').exists()
    rgba=np.array(Image.open(exported/'annotations/train/SOFTWARE_ONLY_0.png'));np.testing.assert_array_equal(rgba[...,0]>0,mask==255);np.testing.assert_array_equal(rgba[...,1]>0,mask==0);assert (rgba[:3,:,:]==0).all()
    checks.append('official_RootPainter_annotation_channels_and_test_exclusion')
    for name in ['unet','deeplabv3plus']:
        out=root/name
        pilot.train(SimpleNamespace(kit=kit,model=name,encoder_weights='none',seed=17,threads=2,output=out,lr=1e-3,epochs=1,batch_size=2,patience=1))
        run=json.loads((out/'training_run.json').read_text());assert run['test_images_used_for_training']==0 and run['epochs_completed']==1
        preds=out/'predictions';pilot.predict(SimpleNamespace(kit=kit,checkpoint=out/'best.pt',output=preds,split='test',threads=2))
        assert np.load(preds/'SOFTWARE_ONLY_3.npy').shape==(64,64)
        pilot.evaluate(SimpleNamespace(kit=kit,predictions=preds,model=name,split='test',threshold=.5,output=out/'evaluation'))
        result=json.loads((out/'evaluation/metrics.json').read_text());assert result['root_length_agreement'] is None and result['manual_length_pairs']==0
        checks.append(name+'_forward_backward_checkpoint_reload_and_prediction_evaluation_software_fixture')
    for row in rows:
        (kit/'labels'/(row['image_id']+'.png')).unlink();(kit/'labels'/(row['image_id']+'.json')).unlink()
    try:pilot.train(SimpleNamespace(kit=kit,model='unet',seed=17,threads=2))
    except ValueError as e:assert 'Missing human mask' in str(e)
    else:raise AssertionError('Training must refuse missing human truth')
    checks.append('training_refuses_absent_human_annotation_before_model_initialization')
pilot.save_json(Path(__file__).parent/'software_validation.json',dict(status='passed',test_kind='software_only_synthetic_fixtures_not_experimental_root_metrics',checks=checks,actual_B3_models_trained=0,actual_B3_human_masks=0,actual_B3_metric_results=None))
print('SOFTWARE_CHECKS_PASS',len(checks))
