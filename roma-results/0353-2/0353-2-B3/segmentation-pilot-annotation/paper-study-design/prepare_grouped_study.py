#!/usr/bin/env python3
"""Draft tube-disjoint study; no model training, pseudo truth or label edits."""
import argparse,csv,hashlib,json,shutil,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
TEST={'0353-3','A017-3','Q1-2'}
VAL=[{'0353-1','A017-1'},{'0353-2','Q1-1'},{'A017-2','Q1-3'}]

def main():
 a=argparse.ArgumentParser();a.add_argument('--inventory',type=Path,required=True);a.add_argument('--annotation-run',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 inventory=json.loads(args.inventory.read_text());positions=[];frames=[];tubes={}
 for x in inventory:
  pos=x['position'];tube=pos.rsplit('-',1)[0];group='tube_holdout_proposal' if tube in TEST else 'development';periods=x['available_periods'];verified=pos=='0353-2-B3'
  row=dict(position_id=pos,tube_id=tube,available_periods=';'.join(periods),missing_periods=';'.join(x['missing_periods']),n_periods=len(periods),proposed_group=group,
   verified_manual_upper_images=len(periods) if verified else 0,annotation_status='verified_user_manual_upper' if verified else 'not_verified',
   fold_0='holdout' if tube in TEST else ('val' if tube in VAL[0] else 'train'),fold_1='holdout' if tube in TEST else ('val' if tube in VAL[1] else 'train'),fold_2='holdout' if tube in TEST else ('val' if tube in VAL[2] else 'train'))
  positions.append(row);tubes.setdefault(tube,[]).append(row)
  for t in periods:frames.append(dict(position_id=pos,tube_id=tube,period=t,proposed_group=group,manual_label_verified=verified,
   source_RGB_reference='0353-2-B3_v9_training_data.zip!0353-2-B3_three_model_v9/data/dataset/upper/'+t+'_RGB.png' if verified else '',
   ROI_coordinates='',manual_mask_reference='0353-2-B3_v9_training_data.zip!0353-2-B3_three_model_v9/data/dataset/upper/'+t+'_mask.png' if verified else '',split_state='PROPOSAL_NOT_TRAINED'))
 assert len(tubes)==9 and len(positions)==19 and len(frames)==68
 assert len({r['position_id'] for r in positions})==19 and len({(r['position_id'],r['period']) for r in frames})==68
 for fold,val in enumerate(VAL):
  train=set(tubes)-TEST-val;assert not (train&val or train&TEST or val&TEST);assert len(train)==4 and len(val)==2
 assert set.union(*VAL)==set(tubes)-TEST and sum(map(len,VAL))==len(set(tubes)-TEST)
 csvout(out/'position_grouping_proposal.csv',positions);csvout(out/'available_images_proposal.csv',frames)
 tuberows=[]
 for tube,items in sorted(tubes.items()):
  tuberows.append(dict(tube_id=tube,positions=len(items),available_date_images=sum(x['n_periods'] for x in items),proposed_group=items[0]['proposed_group'],verified_manual_images=sum(x['verified_manual_upper_images'] for x in items)))
 csvout(out/'tube_inventory.csv',tuberows);foldrows=[]
 for f,val in enumerate(VAL):
  for role,subset in [('train',set(tubes)-TEST-val),('val',val),('holdout',TEST)]:
   pp=[r for r in positions if r['tube_id'] in subset];foldrows.append(dict(fold=f,role=role,tubes=';'.join(sorted(subset)),n_tubes=len(subset),n_positions=len(pp),n_available_date_images=sum(r['n_periods'] for r in pp),n_verified_manual_images=sum(r['verified_manual_upper_images'] for r in pp)))
 csvout(out/'three_fold_proposal.csv',foldrows)
 source=json.loads((args.annotation_run/'original_human_progress.json').read_text());audit=[]
 for item in source['images']:
  v=item['layers']['mask'];mask=np.repeat(np.asarray(v[::2],np.uint8),v[1::2]).reshape(item['height'],item['width'])[:1108];values,counts=np.unique(mask,return_counts=True)
  audit.append(dict(image=item['name'],pixels_by_original_value={str(int(k)):int(v) for k,v in zip(values,counts)},unpainted_value64_fraction=float((mask==64).mean()),
   interpretation='Previous v5-v9 binary targets interpreted upper unpainted64 as background under user completeness declaration; formal ground-truth QA must verify this assumption. Source annotation is unchanged.'))
 js(out/'annotation_representation_audit.json',dict(original_progress_SHA256=sha(args.annotation_run/'original_human_progress.json'),source_annotation_edited=False,images=audit))
 summary=dict(status='DESIGN_PROPOSAL_ONLY_NOT_NEW_TRAINING_RESULTS',inventory_positions=19,tube_groups=9,available_position_date_records=68,
  position_period_counts=dict(Counter(str(x['n_periods']) for x in positions)),development_tubes=6,development_positions=12,development_date_images=42,
  proposed_heldout_tubes=sorted(TEST),heldout_tubes=3,heldout_positions=7,heldout_date_images=26,heldout_period_records=dict(Counter(r['period'] for r in frames if r['tube_id'] in TEST)),heldout_is_same_experiment_not_external_site=True,
  verified_annotated_tubes=1,verified_manual_date_images=4,verified_manual_heldout_images=0,
  split_rule='User confirmed multiple tubes with several positions each; tube IDs parsed before final hyphen. All positions/available periods of each tube stay together. Merge higher plant/pot/plot unit if shared.',
  holdout_choice='One tube per recorded leading-code stratum, with all4 periods represented within every heldout tube. Q1-2 supplies T4 absent from Q1-3. Metadata only, no model score/image difficulty selection; leading codes have no asserted treatment meaning.',
  old_B3_models_reusable_for_CV=False,reason='Fold1 validates0353-2; old root networks trained on0353-2-B3 would leak validation labels. Initialize each fold independently from documented general pretraining.',
  missing_period_policy='No invented frames or negative masks; available images remain useful for binary segmentation. Incomplete observation intervals must be named explicitly in later temporal studies.',
  training_ready=False,blocking_prerequisite='Verified manual labels currently cover only0353-2-B3; multi-tube development/validation/test ROI annotation and QA are needed.',source_inventory_SHA256=sha(args.inventory))
 js(out/'study_design_audit.json',summary);shutil.copyfile(args.inventory,out/'source_position_inventory.json');shutil.copyfile(Path(__file__).with_name('protocol_CN.md'),out/'README.md');shutil.copyfile(__file__,out/'prepare_grouped_study.py')
 refs=[dict(title='RootPainter: Deep Learning Segmentation of Biological Images with Corrective Annotation',URL='https://doi.org/10.1111/nph.18387',official_repository='https://github.com/Abe404/root_painter',consulted='Official project README and mini guide',guide='https://github.com/Abe404/root_painter/blob/master/docs/mini_guide.md'),
  dict(title='nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation',journal='Nature Methods',year=2021,URL='https://github.com/MIC-DKFZ/nnUNet',consulted='Official project and train-models guide; usual5fold CV is a workflow reference, not a root-study sample-size rule'),
  dict(title='clDice-a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation',venue='CVPR',year=2021,URL='https://github.com/jocpae/clDice',consulted='Official repository abstract and loss documentation')]
 js(out/'references.json',dict(consulted_client_date='2026-10-18',timezone='Asia/Shanghai',references=refs,scope='Protocol recommendations combine source methods with this dataset audit; no claim that these papers used the proposed9tube split.'))
 bundle=out/'root_segmentation_paper_study_design.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(out.iterdir()):
   if f.is_file() and f!=bundle and f.name!='SHA256.json':z.write(f,'root_segmentation_paper_study_design/'+f.name)
 with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
 js(out/'SHA256.json',dict(file=bundle.name,bytes=bundle.stat().st_size,SHA256=sha(bundle),CRC_verified=True));print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
