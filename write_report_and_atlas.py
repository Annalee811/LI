from pathlib import Path
import csv,json,io
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import fitz
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill
from docx import Document
from docx.shared import Pt,Mm,RGBColor
from docx.oxml.ns import qn
W=Path(__file__).resolve().parent
F=Path('/workspace/work/root_workflow_real_assets_20261109/fonts')
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def font(n,bold=False):return ImageFont.truetype(str(F/('Times New Roman Bold.ttf'if bold else'Times New Roman.ttf')),n)
CAP1='''Figure 1. Temporal baseline re-evaluation with two root classes. Existing roots combine original persistent and extending labels; New retains the original newly observed label. Ground-truth and predicted masks were grouped before calculating confusion matrices, rather than averaging component-class scores. (A) Existing- and New-class Dice. (B) Unweighted two-class macro Dice and macro IoU. (C) Mean class-specific Precision and Recall. Retrained configurations use three fixed optimization seeds; frozen V66 is one historical reference and has no seed variability estimate. The current-frame classifier collapsed to existing-only predictions under the original budget and is a negative control, not a fully tuned strong single-frame benchmark; its New Precision is undefined. Scores use 83,478 common eligible held-out typed root pixels at T2–T4 in B3, with fixed foreground and background excluded. All original checkpoints and their original three-class validation-based selection are retained; no two-class retraining or checkpoint reselection was performed. Upstream priors had historical B3 exposure. Error bars are optimization-seed sample SD, not biological replication or confidence intervals. Two-class macro scores must not be interpreted as improvements over three-class macro scores.'''
CAP2='''Figure 2. Two-class module re-evaluation and paired structural decoding. (A) Usage of ConvLSTM, GAT and learned message-passing identity gates. Removing ConvLSTM retains other temporal features and graph edges. Removing learned gates retains GAT attention and fixed geometric priors; n/a indicates that message-passing gates are unused without GAT. Common auxiliary relation heads remain present. (B) Seed-matched differences in two-class macro Dice between Full and each ablated configuration (percentage points). Black points are individual seeds; error bars are sample SD of paired seed differences. Negative differences favor the ablated configuration under this fixed protocol. (C, D) Separate paired whole-field structural decoding using identical neural outputs within each seed: raw head, axis constraints, and axis plus junction constraints. Panel C shows two-class macro Dice and panel D shows the two class Dice values. The upper and lower rows have different inference contexts and are not one continuous before/after protocol. The class regrouping preserves New pixels and root identities; class performance alone does not establish recovery of correct root-instance identities.'''
CAP3='''Figure 3. Locked cross-tube prediction cohort and two-class transfer examples. (A) Coverage of three tubes, seven positions and 26 observed images. The complete training tube 0353-2 and the coarse-location position Q1-1-Z2 are excluded. Open squares indicate baseline observations; filled squares indicate post-baseline images; dashes indicate missing acquisitions. Nineteen temporal observations require human two-class ground truth before accuracy evaluation. (B) Same-crop T1–T4 RGB images (left half of each tile) and two-class predictions from the retrained Full neural classification head, using the first fixed seed 20261123 (right half). This is the raw head, not the complete external axis/junction decoder. Example positions are the first canonical position per locked tube. Crops maximize common frozen-foreground coverage over T2–T4 without human labels, confidence selection or prediction-agreement ranking. Gold denotes New and blue Existing; gray baseline roots are unassigned. Bars show 50 native pixels. Images had prior unlabelled prediction exposure, and prepared transfer features retain documented approximations. Human temporal GT is currently unavailable: no independent Dice, IoU, Precision, Recall or generalization accuracy is reported. Frozen V90 deployment-adapter predictions and all retrained head predictions are supplied separately in the accompanying atlas and native-mask bundle.'''
CN='''本次将既有时序基线、模块消融及跨管预测统一按“新生根／既有根”重新评价，其中既有根合并原持续类和延伸类。评价在真值与预测上分别完成类别合并后重算混淆矩阵，未相加或平均原分类评分；模型权重和原检查点选择均不改变，本次并未按两类目标重新训练网络。内部评价仍采用B3共同留出根像素，背景不计入，不能作为独立微根管泛化准确率。'''
RESULT='''两类内部结果中，完整配置宏平均Dice为89.21±0.11%，冻结V66参照为89.21%，当前帧负对照为45.93%；完整配置与冻结时序先验接近，不能据此宣称明显优于该基线。完整配置的新生类Dice为81.63%，既有类为96.79%。相对于移除相应模块的种子配对结果，保留ConvLSTM、GAT和学习式身份门控的平均宏Dice差值分别为−0.056、+0.105和+0.065个百分点；ConvLSTM未表现出正向平均增益，GAT也并非每个种子都改善，因此本轮不能认定所有模块均有稳定收益。在另一个配对完整视野结构解码实验中，同一神经输出的两类宏Dice由89.12±0.10%提高至根轴约束后的92.00±0.08%，并在加入连接处约束后达到92.41±0.34%；对应的新生与既有Dice最终为87.09%和97.73%。该变化属于同一输出的结构解码干预，不应与分区隔离内部主表拼作一条连续优化曲线。类别合并消除了持续／延伸之间的混淆，因此不能把较高的两类宏Dice与原三类宏Dice直接比较，也不能把类别正确率等同于根轴归属正确率。'''
LIMIT='''独立部分锁定0353-3、A017-3、Q1-2三个微根管，共7个位置26幅影像，含7个基准期和19个后续观测。所有已运行的390张重训练模块预测与52张固定参照均已映射为两类，共442张原生掩膜。主图展示固定首个种子的Full原始分类头；其外部输出尚不包含本次B3配对实验所用的完整根轴／连接处解码。原冻结V90部署适配预测另在全部7个位置的图集中列出，不与重训练Full头混称同一推理协议。目前无跨管人工时序真值，独立P/R、Dice、IoU保持未计算，未用模型预测或全255空模板代替真值。新增真值需明确标记背景0、既有1、新生2、不确定255；此前仅标“新生／非新生”的二值真值不能直接区分既有根与背景。三次随机种子只反映优化变异，不是生物学重复；未生成P值。'''
def atlas():
 rows=read(W/'Independent_manifest.csv');doc=fitz.open()
 for pos in sorted(set(r['position_id']for r in rows)):
  rr={r['period']:r for r in rows if r['position_id']==pos};canvas=Image.new('RGB',(2600,1500),'white');d=ImageDraw.Draw(canvas)
  d.text((35,20),pos+' | Two-class cross-tube predictions',font=font(42,True),fill='black');d.text((35,78),'Human GT unavailable: all independent accuracy metrics remain uncomputed.',font=font(29),fill='black')
  models=[('Original RGB',None),('Frozen V66',W/'independent_two_class_predictions/V66_pairwise_frozen/frozen'),('Retrained Full head',W/'independent_two_class_predictions/full/20261123'),('Frozen V90 transfer',W/'independent_two_class_predictions/V90_transfer_reference/frozen')]
  for j,t in enumerate(['T1','T2','T3','T4']):d.text((525+j*565,125),t,font=font(31,True),fill='black')
  for i,(label,folder)in enumerate(models):
   y=180+i*285;d.text((22,y+100),label,font=font(25,True),fill='black')
   for j,t in enumerate(['T1','T2','T3','T4']):
    x=300+j*565
    if t not in rr:d.text((x+140,y+110),'Not acquired',font=font(29),fill='black');continue
    rgb=np.array(Image.open(rr[t]['RGB_path']).convert('RGB'))
    if folder:
     p=np.array(Image.open(folder/pos/(t+'_two_classes.png')))
     for code,color in [(1,(63,112,145)),(2,(197,141,49)),(255,(205,205,205))]:
      v=p==code;rgb[v]=np.rint(.90*np.array(color)+.10*rgb[v]).astype(np.uint8)
    im=Image.fromarray(rgb);im.thumbnail((548,260),Image.Resampling.LANCZOS);canvas.paste(im,(x+(548-im.width)//2,y+(260-im.height)//2))
  for x,color,label in [(450,(197,141,49),'New'),(1040,(63,112,145),'Existing')]:d.rectangle((x,1340,x+34,1364),fill=color);d.text((x+48,1331),label,font=font(29,True),fill='black')
  d.text((35,1400),'Full head: first fixed seed 20261123. Frozen V90: separate historical model and documented transfer adapter.',font=font(25),fill='black')
  d.text((35,1440),'Gray roots are baseline/unassigned. Missing dates stay absent. These panels compare predictions, not GT-based accuracy.',font=font(25),fill='black')
  b=io.BytesIO();canvas.save(b,format='JPEG',quality=96,subsampling=0);p=doc.new_page(width=780,height=450);p.insert_image(p.rect,stream=b.getvalue())
 doc.save(W/'Independent_all_7_positions_two_class_atlas.pdf',deflate=True);doc.close()
def main():
 (W/'Results_and_limitations_CN.txt').write_text(CN+'\n\n'+RESULT+'\n\n'+LIMIT+'\n')
 (W/'Figure_captions_EN.txt').write_text(CAP1+'\n\n'+CAP2+'\n\n'+CAP3+'\n')
 checklist=read(W/'Independent_manifest.csv')
 for r in checklist:r['annotation_schema']='background_0_existing_1_new_2';r['temporal_scoring_required']=bool(r['previous_observed_period'])
 with(W/'Independent_two_class_annotation_checklist.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(checklist[0]));w.writeheader();w.writerows(checklist)
 wb=Workbook();wb.remove(wb.active)
 for title,name in [('Mean_SD','Two_class_mean_SD.csv'),('Internal_each_run','Internal_two_class_runs.csv'),('Classwise_each_run','Internal_two_class_classwise.csv'),('Paired_effects','Paired_module_two_class_effects.csv'),('Decoder_each_seed','Paired_decoder_two_class.csv'),('Decoder_classwise','Paired_decoder_two_class_classwise.csv'),('Frozen_references','Historical_two_class_references.csv')]:
  rr=read(W/'metrics'/name);ws=wb.create_sheet(title);keys=list(rr[0]);ws.append(keys)
  for r in rr:
   values=[]
   for key in keys:
    v=r[key]
    if v=='':v=None
    else:
     try:v=float(v)if any(c in v for c in '.eE')else int(v)
     except(ValueError,TypeError):pass
    values.append(v)
   ws.append(values)
  for i,key in enumerate(keys,1):
   if any(k in key for k in ['Dice','IoU','Precision','Recall'])and not key.endswith('_pp'):
    for rr0 in ws.iter_rows(min_row=2,min_col=i,max_col=i):rr0[0].number_format='0.00%'
  for c in ws[1]:c.font=Font(name='Times New Roman',bold=True,color='000000');c.fill=PatternFill('solid',fgColor='E4EBF0')
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(48,max(14,len(str(col[0].value))+2))
  ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
 ws=wb.create_sheet('Independent_status');ws.append(['Class','Precision','Recall','Dice','IoU','Status'])
 for c in ['Existing','New']:ws.append([c,None,None,None,None,'NOT_EVALUATED_NO_HUMAN_GT'])
 for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=26
 ws=wb.create_sheet('Independent_cohort');keys=list(checklist[0]);ws.append(keys)
 for r in checklist:ws.append([r[k]for k in keys])
 ws.freeze_panes='C2'
 wb.save(W/'Three_key_figures_two_class_metrics.xlsx')
 doc=Document();s=doc.sections[0];s.left_margin=Mm(15);s.right_margin=Mm(15);s.top_margin=Mm(16);s.bottom_margin=Mm(16)
 st=doc.styles['Normal'];st.font.name='Times New Roman';st.font.size=Pt(10);st.font.color.rgb=RGBColor(0,0,0);st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 for i,(stem,caption)in enumerate([('Figure_1_Temporal_baselines_two_classes',CAP1),('Figure_2_Module_ablations_two_classes',CAP2),('Figure_3_Independent_transfer_two_classes',CAP3)]):
  if i:doc.add_page_break()
  doc.add_picture(str(W/(stem+'.png')),width=Mm(175));doc.add_paragraph(caption)
 doc.add_page_break();doc.add_paragraph('结果与解释').runs[0].bold=True
 for text in [CN,RESULT,LIMIT]:doc.add_paragraph(text)
 doc.save(W/'Three_key_figures_captions_and_results.docx')
 (W/'README.txt').write_text('''THREE KEY FIGURES — TWO ROOT CLASSES (2026-11-23)

Existing = original persistent1 + extending3; New = original newly observed2.
This is a post-hoc two-class re-evaluation of the existing15 trained runs. Model
weights, original three-class training objectives and validation checkpoint choices
were not changed. No new two-output network is claimed to have been trained.
Class masks and truth were merged before metrics. Do not average old class scores.
New-class metrics remain identical; two-class macros cannot be compared directly
with three-class macros as if the change were a model improvement.

Figure1: current-frame negative control, frozen V66 pairwise reference,
retrained ConvLSTM without GAT, and retrained Full temporal graph. Internal B3
eligible root support only, background excluded. Frame-only collapsed to all
Existing, so New Precision is undefined. This is not a tuned strong single-frame
benchmark. The frozen prior had historical B3 exposure; this is conditional
internal evaluation, not independent generalization.

Figure2: matched module usage/effects plus a SEPARATE paired whole-field decoder
experiment. Three fixed seeds measure optimization variation, not biological n.
No p-values or significance claims are made. Memory removal retains other
historical features; gate removal retains GAT attention/geometric priors.
Grouping preserves root IDs; good class Dice does not prove correct root identity.

Figure3: locked3 tubes,7 positions,26 input images; training tube0353-2 excluded.
The19 post-baseline images still lack human temporal truth. All independent
metrics are NULL, not zero. Main examples use the raw retrained Full neural head
at the first fixed seed20261123; the complete axis/junction decoder is not
included in these external head predictions. The separate frozen V90 transfer
adapter (historical weights and documented input approximations) is shown in
the7-page full-position atlas, alongside V66 and the retrained head.
Prior unlabelled image/prediction exposure is disclosed; not a prospective
never-seen-image trial. Example positions are first canonical IDs per tube;
crop selection uses common binary foreground coverage, never GT or quality rank.

Files: three600dpi JPG/PNG/TIFF figures, font-embedded PDF, editable SVG,
Word captions/results, Excel full values, and442 native two-class predictions
(390 retrained head maps +52 frozen-reference maps). Source models/labels untouched.
Colors: gold New, blue Existing; gray baseline is unassigned, not a third root type.
T1 not scored; missing dates not synthesized. Original class2 pixels unchanged.

Numeric reproduction (NumPy,Pillow):
python reproduce_metrics.py --bundle . --output /new/empty/check
Saved held-out truth and binary predictions reproduce all26 confusion matrices.
The original15 trained best/resume checkpoints, logs and source are still at:
https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/V90_actual_experiments_code_weights_results.zip
Checkpoint hashes are retained in Existing_checkpoints.json.

Future independent evaluation:
Use Independent_two_class_annotation_checklist.csv. From RGB independently,
annotate0 background,1 Existing,2 New,255 uncertain. A New-vs-non-New binary
truth cannot distinguish Existing from background and must not be reused as such.
Set annotation_state=human_complete and fill annotator,annotation_source.
Keep annotation_schema=background_0_existing_1_new_2. Then:
python evaluate_independent_two_class.py --manifest completed.csv \
 --annotation-root /GT --prediction-dir /one/model/seed --output /new/evaluation
The full26-image cohort is locked; T1 is excluded from temporal scores. Ground
truth background IS included in independent class errors, unlike the internal
fixed-root evaluation. Score raw grouped predictions, not masks with rejected
pixels silently removed. Empty GT and missing human provenance are rejected.
Original3-class human labels can also be grouped with --gt-encoding original-three
and explicit schema background_0_persistent_1_new_2_extension_3.

All figure text is black Times New Roman, white background,8pt axes/12pt letters.
To redraw: python draw_three_key_figures.py --fonts /licensed/font/directory
Fonts not redistributed. Standard dependencies: numpy,Pillow,matplotlib,
PyMuPDF,openpyxl,python-docx. Dataset/current identity policy is unchanged.
''')
 atlas();print('THREE_FIGURE_TABLES_CAPTIONS_AND_ATLAS_COMPLETE',flush=True)
if __name__=='__main__':main()
