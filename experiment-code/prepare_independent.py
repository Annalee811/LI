from pathlib import Path
import csv,json,hashlib,sys,shutil
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parents[1];sys.path.insert(0,'/workspace/LI/analysis-scripts');from dataset_identity import require_active_position
rows=list(csv.DictReader(open('/workspace/LI/dataset/current/active_images.csv',encoding='utf-8-sig')));testtubes={'0353-3','A017-3','Q1-2'};records=[];kit=W/'independent_annotation_kit';kit.mkdir(exist_ok=True)
for pos in dict.fromkeys(r['position_id']for r in rows):
 require_active_position(pos);prev=None
 for r in [q for q in rows if q['position_id']==pos]:
  tube=r['tube_id'];role='excluded_training_tube'if tube=='0353-2'else'excluded_coarse_localization'if pos=='Q1-1-Z2'else'locked_independent_test'if tube in testtubes else'eligible_annotation_pool';gt=kit/pos/(r['period']+'_GT.png');RGB=Path('/workspace/work/all_series_RGB_PDF_20261105/sources')/r['canonical_view_relative_path'];assert hashlib.sha256(RGB.read_bytes()).hexdigest()==r['registered_RGB_SHA256']
  if role=='locked_independent_test':
   folder=kit/pos;folder.mkdir(exist_ok=True);shutil.copy2(RGB,folder/(r['period']+'_RGB.png'))
   if not gt.exists():Image.fromarray(np.full((int(r['RGB_height']),int(r['RGB_width'])),255,np.uint8)).save(gt)
  records.append(dict(position_id=pos,tube_id=tube,period=r['period'],previous_observed_period=prev or'',role=role,RGB_SHA256=r['registered_RGB_SHA256'],width=r['RGB_width'],height=r['RGB_height'],RGB_path=str(RGB),GT_path=str(gt)if role=='locked_independent_test'else'',annotation_state='pending',annotator='',annotation_source='',input_quality_flag=r['review_required'],prior_unlabelled_prediction_exposure=True));prev=r['period']
for name,rr in [('independent_manifest.csv',records),('annotation_checklist.csv',[r for r in records if r['role']=='locked_independent_test'])]:
 with(W/'reports'/name).open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
shutil.copy2(W/'reports/annotation_checklist.csv',kit/'annotation_checklist.csv');protocol=dict(locked_test_tubes=sorted(testtubes),locked_test_positions=len(set(r['position_id']for r in records if r['role']=='locked_independent_test')),locked_test_frames=sum(r['role']=='locked_independent_test'for r in records),training_tube_excluded='0353-2',all_eligible_cross_tube_frames=sum(r['role']in ['locked_independent_test','eligible_annotation_pool']for r in records),coarse_localization_excluded='Q1-1-Z2',selection_basis='Predeclared tube IDs spanning all three germplasm series; no test-label-based selection',test_images_had_prior_unlabelled_prediction_exposure=True,unseen_image_or_prospective_blind_test_claim=False,status='NOT_EVALUATED_NO_HUMAN_GT',metrics=None,annotation_instructions='GT is human annotation only, not model predictions; templates contain 255 (unannotated), not zero-valued background truth',locked_before_test_labels_available=True)
(W/'reports/independent_protocol.json').write_text(json.dumps(protocol,indent=2));(kit/'README.txt').write_text('独立测试人工真值待标注材料（不是分割结果）\n\n选定0353-3、A017-3、Q1-2三个整管，共7个位置26幅图像。0353-2整管不进入独立测试。\nRGB为现有配准图像的原字节副本；未重新配准、未改变尺寸。\nGT模板目前全为255，表示尚未标注，不能当作已完成真值或全背景。\n编码：0背景，1持续，2新出现，3延伸，4首期基准根，255不确定/忽略。\n同一位置要联合查看所有实际日期；缺失日期不补图。变化相对于previous_observed_period字段。\n请独立依据RGB标注，不把模型预测复制为GT。完成后修改annotation_state=human_complete，并填写annotator和annotation_source。\n三类准确率测试在真值齐备前保持未计算。图像此前做过无标签预测，不宣称从未见过影像的前瞻盲测。\n');print(json.dumps(protocol,indent=2))
