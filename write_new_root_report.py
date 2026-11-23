from pathlib import Path
import csv,json,io,shutil
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
CN='''本次结果展示聚焦于新生根，仅统计精确率（Precision）和召回率（Recall）。新生根直接取自原时序类别2，其余有效像素作为非新生像素参与误检和漏检统计，未合并其他类别、未修改人工标注或重新训练模型。在B3全图人工有效像素的历史一致性评价中，V66的新生精确率和召回率分别为85.03%和85.32%，V90分别为94.51%和84.50%；V90的新生误报像素由11,831降至3,869，但漏检像素由11,561增至12,208，表现为精确率提高9.47个百分点、召回率下降0.82个百分点。因此，该结果支持新生误报减少，不能表述为精确率和召回率同时改善。另在冻结上游先验的内部模块控制实验中，完整新增配置的新生精确率和召回率为82.68±0.59%和80.60±0.25%。在单独开展的配对完整视野结构解码实验中，同一神经输出的新生精确率由82.39±0.48%增至87.76±0.46%，召回率由80.59±0.14%增至86.43±0.91%。这些统计分别对应全图历史一致性、分区隔离内部控制和完整视野配对解码三种协议，不应混作一个连续提升过程。误差表示三个优化种子的标准差，不是生物学重复。18个位置共66幅图像已导出新生掩膜、原图叠加及根像素提取图，其中B3使用原自动V90输出，其余位置使用既有V90部署适配预测。其他微根管缺少人工时序真值，其新生精确率和召回率仍未计算。首期仅作为基准；缺失时期不补图，变化相对于前一个实际观测时期。'''
EN='''Reporting was restricted to newly observed roots, using Precision and Recall as the only performance endpoints. Original temporal class 2 was treated as positive, and all other human-valid pixels remained negative; no classes, annotations, or model weights were altered. For descriptive historical agreement over the full B3 image domain, V66 achieved 85.03% Precision and 85.32% Recall, whereas V90 achieved 94.51% and 84.50%, respectively. False-positive new-root pixels decreased from 11,831 to 3,869, but false-negative pixels increased from 11,561 to 12,208. Thus, the higher Precision was accompanied by a small reduction in Recall. In the separate conditional internal module study, the full newly trained configuration achieved 82.68 ± 0.59% Precision and 80.60 ± 0.25% Recall. In the paired whole-field decoding study, using identical neural outputs within each optimization seed, Precision increased from 82.39 ± 0.48% to 87.76 ± 0.46%, and Recall increased from 80.59 ± 0.14% to 86.43 ± 0.91%. The historical full-image, partition-isolated module, and paired whole-field decoding protocols should be interpreted separately. Standard deviations represent variation over three optimization seeds, not biological replication. New-root masks and RGB overlays were exported for 66 images at 18 positions; B3 uses the original automatic V90 output, whereas other positions use the documented V90 transfer adapter. Cross-tube Precision and Recall remain unmeasured because human temporal ground truth is unavailable. Baseline images are not scored for emergence, and changes across missing dates refer to the previous actually observed date.'''
CAP1='''Figure X. Newly observed root Precision and Recall in controlled internal experiments. (A, B) Precision and Recall on common eligible held-out B3 typed root pixels, conditioned on the frozen foreground and upstream priors. Frozen V66 is one historical reference; retrained configurations use three optimization seeds. Frame-only Precision is undefined because no pixels were predicted positive, and is not replaced by zero. (C, D) Paired whole-field structural decoding results using identical neural outputs within each seed. Points and error bars show means and sample standard deviations over optimization seeds; thin gray lines connect matched seeds. The two rows use distinct inference contexts. Only original temporal class 2 is positive; other valid root classes remain negative. Background is excluded from this internal typed-root domain. No independent cross-tube accuracy or statistical significance is inferred.'''
CAP2='''Figure Y. Historical V66–V90 comparison restricted to newly observed roots. (A, B) Precision and Recall pooled over human-valid full-image pixels, including background, at T2–T4 in B3. Pixels marked uncertain in the human truth are excluded. These are descriptive historical agreement scores for one sequence, not held-out independent accuracy estimates; no error bars are assigned. (C) Original RGB, human new-root annotations, and automatic V66 and V90 new-root predictions in the same crop across T1–T4. Only newly observed roots are highlighted in gold; no other class is merged into this class. The crop is selected using human new-root coverage across dates, without prediction-quality selection. The scale bar represents 100 native pixels. T1 is a baseline without an emergence target. Crop coordinates and input hashes are retained separately.'''
README='''新生根单类结果包 / Newly observed roots only — 2026-11-23

这次是从已有模型输出中提取新生类并重算/导出新生Precision与Recall，未重新训练模型。
原多类模型、原始图像、人工标注、权重和旧结果均保留。这里只展示新生根，不把延伸段等其他类别合并为新生。
新生定义：原时序类别2。其他有效像素仍作为非新生参与误检、漏检统计，防止缩小分母。
Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。无预测正像素时Precision未定义，不填0。

结果协议分别为：
1. B3全图历史人工一致性：T2–T4合并，包含人工有效背景，忽略255不确定像素。
2. 分区隔离内部模块对照：共同的有效留出根像素；固定上游前景，不包含背景。
3. 配对完整视野结构解码：同一检查点的相同神经输出，另行比较解码阶段。
请分别使用三类协议，不拼接为一条连续优化曲线，不把B3历史一致性称为跨管准确率。
既有检查点选择保持原样，本次没有按新指标重新选权重或改测试集。

主图：B3_V66_V90_new_roots_only_SCI.*（P、R及同视野T1–T4的RGB/人工/V66/V90）
实验图：New_root_precision_recall_SCI.*（新生P、R的内部控制与配对解码）
字体Times New Roman、黑字白底；位图600 dpi，PDF嵌入字体，SVG可编辑。
New_root_results_and_captions.docx：可编辑中英文结果及英文图注。
New_root_Precision_Recall.xlsx：只包含新生类P、R及其计数、清单，不输出其他类别指标。

All_18_series_new_roots_only.pdf：18个位置、66幅图像，每页原RGB/新生叠加/新生黑白掩膜。
每管ZIP里的每个位置均提供：
  Tn_new_mask.png：原生尺寸，0非新生、255新生。
  Tn_new_RGB_overlay.png：仅新生像素着色，其他原始RGB像素逐值不变。
  Tn_new_root_RGB_on_white.png：新生根的原RGB像素，其他像素白色。
  Tn_new_ownership.npz：仅保留新生位置的原axis/track_id，其他位置-1；不强制填补未知归属。
  New_root_T1-T4.jpg：该位置的四期合成页；未采集日期留空并标Not acquired。
T1是基准，空新生掩膜不代表根系没有发生或真实新生率为0。缺失时期不合成；变化参照前一个实际日期。
Q1-1-Z2的T2/T3为粗定位，按描述性预测展示，不进入主独立测试。Q1-3-Z3排除。

B3使用原自动V90输出；其他位置使用已记录、含近似的V90部署适配结果。
除B3外尚无人工时序真值，不计算跨微根管P或R，不将预测当人工真值。
基线/消融预测单类包另含442张新生二值掩膜（390张新训练输出、52张固定参照）。

未来独立评价：沿用固定3管、7位置、26幅输入，其中7个基准期不评新生，19个后续影像需人工真值。
独立输入名单保持不变。在清单中填写human_complete、annotator、annotation_source，标签来自RGB。
python evaluate_new_roots.py --manifest 完成清单.csv --annotation-root /path/to/GT \
 --prediction-dir /path/to/one/model/seed --output /new/empty/evaluation
默认GT为原时序标签0/1/2/3/255（2新生），预测为0/255新生掩膜。
也可用--gt-encoding new-binary接受Tn_new_GT.png：0非新生、1新生、255不确定。
空模板/无人工来源会拒绝评分。按整管汇总，不把日期或小图当作独立生物重复。

Source scripts include explicit source paths in the current environment. Override the module
constants W/STUDY/O/REF/RGBROOT/F if reproducing elsewhere. Standard dependencies:
numpy, Pillow, matplotlib, PyMuPDF, openpyxl, python-docx. No Torch is needed for this derivation.
'''
def independent_preview():
 rows=read(W/'Independent_new_root_annotation_checklist.csv');doc=fitz.open();dest=W/'independent_new_only_previews';dest.mkdir(exist_ok=True)
 for pos in sorted(set(r['position_id']for r in rows)):
  rr={r['period']:r for r in rows if r['position_id']==pos};canvas=Image.new('RGB',(2600,1500),'white');d=ImageDraw.Draw(canvas)
  d.text((35,22),pos+' | Newly observed roots only',font=font(43,True),fill='black')
  d.text((35,80),'Cross-tube predictions; human temporal ground truth unavailable. No Precision or Recall assigned.',font=font(28),fill='black')
  pairs=[('Original RGB',None),('Frozen V66',W/'baseline_ablation_new_masks/V66_pairwise_frozen/frozen'),('New full model',W/'baseline_ablation_new_masks/full/20261123'),('V90 transfer',W/'baseline_ablation_new_masks/V90_transfer_reference/frozen')]
  for j,t in enumerate(['T1','T2','T3','T4']):d.text((525+j*570,127),t,font=font(32,True),fill='black')
  for i,(label,root)in enumerate(pairs):
   y=185+i*285;d.text((22,y+95),label,font=font(27,True),fill='black')
   for j,t in enumerate(['T1','T2','T3','T4']):
    x=290+j*570
    if t not in rr:d.text((x+150,y+105),'Not acquired',font=font(28),fill='black');continue
    rgb=np.array(Image.open(rr[t]['RGB_path']).convert('RGB'))
    if root:
     n=np.array(Image.open(root/pos/(t+'_new_mask.png')))==255;rgb[n]=np.rint(.92*np.array([216,153,49])+.08*rgb[n]).astype(np.uint8)
    im=Image.fromarray(rgb);im.thumbnail((552,260),Image.Resampling.LANCZOS);canvas.paste(im,(x+(552-im.width)//2,y+(260-im.height)//2))
  d.rectangle((35,1350,69,1375),fill=(216,153,49));d.text((85,1341),'Newly observed root',font=font(28,True),fill='black')
  d.text((35,1400),'New full model: fixed seed 20261123, not selected using test performance. V90 uses the documented transfer adapter.',font=font(25),fill='black')
  d.text((35,1440),'T1 is baseline only. Missing dates are not synthesized. All non-new RGB pixels remain unchanged.',font=font(25),fill='black')
  canvas.save(dest/(pos+'.jpg'),quality=96,subsampling=0,dpi=(300,300));b=io.BytesIO();canvas.save(b,format='JPEG',quality=96,subsampling=0);p=doc.new_page(width=780,height=450);p.insert_image(p.rect,stream=b.getvalue())
 doc.save(W/'Independent_new_roots_only_comparison.pdf',deflate=True);doc.close()
def main():
 (W/'New_root_results_CN.txt').write_text(CN+'\n');(W/'New_root_results_EN_and_captions.txt').write_text(EN+'\n\n'+CAP1+'\n\n'+CAP2+'\n');(W/'README.txt').write_text(README)
 wb=Workbook();wb.remove(wb.active)
 inputs=[('Internal_mean_SD',W/'metrics/New_root_internal_mean_SD.csv'),('Internal_each_run',W/'metrics/New_root_internal_each_run.csv'),('Paired_decoder',W/'metrics/New_root_paired_decoder.csv'),('Frozen_heldout',W/'metrics/New_root_frozen_heldout_reference.csv'),('B3_full_image',W/'metrics/New_root_B3_full_image_historical_agreement.csv'),('All_66_prediction_index',W/'All_66_new_root_prediction_manifest.csv'),('Independent_label_list',W/'Independent_new_root_annotation_checklist.csv')]
 for title,path in inputs:
  rr=read(path);ws=wb.create_sheet(title);keys=list(rr[0]);ws.append(keys)
  for r in rr:
   vals=[]
   for key in keys:
    v=r[key]
    if v=='':v=None
    elif key not in ['position_id','tube_id','period','seed','previous_observed_period']:
     try:v=float(v)if any(x in v for x in '.eE')else int(v)
     except(ValueError,TypeError):pass
    vals.append(v)
   ws.append(vals)
  for c in ws[1]:c.font=Font(name='Times New Roman',size=10,bold=True,color='000000');c.fill=PatternFill('solid',fgColor='E5EDF2')
  for row in ws.iter_rows(min_row=2):
   for cell in row:cell.font=Font(name='Times New Roman',size=10,color='000000')
  for i,key in enumerate(keys,1):
   ws.column_dimensions[ws.cell(1,i).column_letter].width=min(48,max(14,len(key)+3))
   if key in ['Precision','Recall','Precision_mean','Recall_mean','Precision_SD','Recall_SD']:
    for row in ws.iter_rows(min_row=2,min_col=i,max_col=i):row[0].number_format='0.00%'
  ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
 ws=wb.create_sheet('Read_me');ws.column_dimensions['A'].width=125
 for line in README.splitlines():ws.append([line])
 wb.save(W/'New_root_Precision_Recall.xlsx')
 doc=Document();sec=doc.sections[0];sec.top_margin=Mm(16);sec.bottom_margin=Mm(16);sec.left_margin=Mm(15);sec.right_margin=Mm(15)
 sty=doc.styles['Normal'];sty.font.name='Times New Roman';sty.font.size=Pt(10);sty.font.color.rgb=RGBColor(0,0,0);sty.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 doc.add_paragraph('Newly observed roots: Precision and Recall').runs[0].bold=True
 doc.add_picture(str(W/'B3_V66_V90_new_roots_only_SCI.png'),width=Mm(175));doc.add_paragraph(CAP2)
 doc.add_paragraph(CN);doc.add_paragraph(EN);doc.add_page_break()
 doc.add_picture(str(W/'New_root_precision_recall_SCI.png'),width=Mm(175));doc.add_paragraph(CAP1)
 doc.save(W/'New_root_results_and_captions.docx')
 # Human binary references use 1=new, 0=other, 255=human-uncertain, preserving ignore pixels.
 ref=Path('/workspace/work/V66_V90_same_field_SCI_20261113/sources/reference');out=W/'B3_human_new_truth';out.mkdir(exist_ok=True)
 for t in ['T2','T3','T4']:
  gt=np.array(Image.open(ref/(t+'_classes.png')));new=(gt==2).astype(np.uint8);new[gt==255]=255;Image.fromarray(new).save(out/(t+'_new_GT.png'))
 independent_preview();print('NEW_ONLY_REPORT_TABLES_AND_PREVIEWS_COMPLETE',flush=True)
if __name__=='__main__':main()
