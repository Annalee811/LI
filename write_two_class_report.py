from pathlib import Path
import csv,json,shutil
import numpy as np
from PIL import Image
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill
from docx import Document
from docx.shared import Pt,Mm,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
W=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def display(v):return '—'if v==''or v is None else f'{float(v):.2f}'
def main():
 rows=read(W/'Series_mean_adjacent.csv');stats={m:{r['period']:r for r in rows if r['material']==m}for m in ['0353','A017','Q1']}
 cn='''后续基于V90预测的根段构成统计统一采用新生根和既有根两类。新生根保持原新生类别不变，既有根由原持续根与延伸根合并；合并仅改变根段类别分组，不改变根轴身份、二元前景或人工标注。各类别长度在同一二元根系骨架上测量：水平/垂直边权重为1，斜边为√2，存在正交连接的冗余斜边被排除，每条边向其两个端点各分配一半长度，孤立骨架像素记为1。由此，新生率为L新/(L持续+L新+L延伸)×100%，既有率为(L持续+L延伸)/(L持续+L新+L延伸)×100%，两者相加为100%。这里的“率”指本期可见预测骨架长度占比，不是根条数比例或每日生长速率；既有根包含原根轴新延长的部分，不能解释为所有像素在上一期均已存在。'''
 desc=[]
 for m in ['0353','A017','Q1']:
  d=stats[m];n=[float(d[t]['new_mean_pct'])for t in ['T2','T3','T4']];e=[float(d[t]['existing_mean_pct'])for t in ['T2','T3','T4']]
  desc.append(f'{m}的新生根平均长度占比在T2、T3和T4分别为{n[0]:.2f}%、{n[1]:.2f}%和{n[2]:.2f}%，既有根对应为{e[0]:.2f}%、{e[1]:.2f}%和{e[2]:.2f}%。')
 cn2=''.join(desc)+'''三个系列的新生根预测占比均在T2最高，既有根占比均在T4最高，表现为可见根段构成逐步转向既有根轴占优势。该结果是可用观察视野的描述性均值；不同阶段可用视野数不同，不能仅据此推断种质间显著差异或根系死亡。'''
 cn3='''本次直接读取既有V90输出，覆盖18个观察位置、8个微根管和66幅影像，未重新训练或配准。B3使用原自动V90结果，其余位置使用已记录、含近似的冻结V90部署适配预测。主图沿用真实相邻时期的视野等权均值，共43个区间；5个跨缺失时期区间单列，另附全部48个前后观测区间的均值、微根管等权均值及粗定位排除敏感性结果。T1不赋时序占比，6个未采集时期保持缺失。各影像先合并类别再计算标准差，未相加原两类标准差；由于两个占比精确互补，同一样本集合中它们的标准差相同。阴影表示视野间样本标准差，不是置信区间，位置和日期不作为独立生物重复。'''
 cn4='''主结果为原始预测类别的长度占比，包含尚带不确定标记的暂定类别，不是经人工真值验证的组成比例。原不确定标记完整保留；另提供只在已分配骨架支持上重新归一化的两类占比及覆盖率，不与主结果混用。B3原自动输出没有同格式的不确定标记时，相关覆盖率保持缺失而不填100%。Q1-1-Z2的T2/T3仍为粗定位，仅作描述性展示。旧版三系列平均率基于V35/V66，本次没有将旧数值改名充作V90结果。其他微根管尚缺人工时序真值，因此本次不提供独立准确率结论。'''
 caption='''Figure X. Two-class composition of V90-predicted visible root skeleton length across four observation stages. Panels show 0353, A017 and Q1. Newly observed roots retain original class 2; existing roots combine persistent and elongating segments (original classes 1 and 3). Fractions are calculated on a single binary skeleton per image using non-redundant eight-neighbour edges and equal endpoint allocation, conserving total predicted skeleton length. Points denote unweighted means of per-position fractions for genuinely adjacent observations; shaded bands show sample SD across available fields, clipped to 0–100% for display. T2–T4 field counts are 6/6/6 for 0353, 6/4/4 for A017, and 4/3/4 for Q1. The data include 18 positions nested within eight tubes; dates and positions are not independent biological replicates. T1 is baseline and has no temporal fractions. Five gap-spanning comparisons are excluded from the main curves and provided separately. Fractions sum to 100% by composition, not by classification accuracy. Existing-root status refers to axis identity and includes newly elongated portions. Raw provisional predictions are summarized; uncertainty-retained maps, assigned-support fractions and coverage are supplied separately. No independent biological or cross-tube accuracy inference is made.'''
 english='''Downstream V90 composition analyses were regrouped into newly observed and existing roots without changing model weights, foreground geometry, root-instance assignments or annotations. Existing roots comprise the original persistent and elongating classes. At each image, class lengths were assigned on the same binary skeleton by distributing half of each unique weighted edge to each endpoint. New-root share was 100L_N/(L_P+L_N+L_E), and existing-root share was 100(L_P+L_E)/(L_P+L_N+L_E). Per-position grouping preceded calculation of sample SD; component SDs were not added. The two fractions are complementary, so their SDs are identical for the same observations. The main summaries retain the historical descriptive convention of equally weighting available positions within each material and stage, using only truly adjacent observed dates. Tube-balanced and gap-inclusive summaries are provided separately. The proportions describe model-derived current visible skeleton length, not root counts, daily growth rates, or independently validated biological traits.'''
 (W/'Two_class_methods_and_results_CN.txt').write_text(cn+'\n\n'+cn2+'\n\n'+cn3+'\n\n'+cn4+'\n')
 (W/'Two_class_methods_and_caption_EN.txt').write_text(english+'\n\n'+caption+'\n')
 readme='''V90 downstream two-class composition — 2026-11-23

Default groups: New = original 2; Existing = original 1 + 3. This is category
grouping after prediction, not retraining a two-output network. Root-axis/track
maps are copied byte-for-byte; root instances are not merged by this grouping.
Existing refers to root-axis identity and includes newly elongated portions.

Rates are current skeleton-length fractions, not pixel-area fractions or root counts.
The same binary skeleton is used for all groups. Non-redundant H/V edges weigh 1,
diagonals weigh sqrt(2), isolated pixels weigh 1; half each edge goes to each endpoint.
L_existing = L_persistent + L_extension; r_new + r_existing = 100% on typed frames.
This conservation is not a claim of 100% predictive accuracy.
T1 has no temporal fractions. Missing dates are NA, not zero. No mm/day conversion.

Primary figure: V90_three_series_two_class_rates_SCI.*
Alternative 3-D figure: V90_three_series_two_class_3D_SCI.*
Main data: Series_mean_adjacent.csv (43 actual adjacent comparisons).
Gap-inclusive data: Series_mean_previous_observed.csv (48 observed intervals).
Five gap comparisons are not mixed into the main adjacent-stage curves.
Tube-balanced, coarse-position-excluded and assigned-support sensitivities are separate.
Main means are per-field arithmetic means, matching the previous descriptive convention;
SD is a sample field SD, not SEM/CI or a count of independent biological replicates.
There are 18 fields nested within eight tubes. Category SDs are recomputed after grouping.
Two complementary percentages have equal SD; this is not a duplicated-data error.

Raw primary shares retain provisional class assignments for all predicted root support.
Original uncertainty flags are retained in alternate maps and in skeleton cache files.
Assigned-support shares use only unflagged support, with a different denominator;
use their coverage and keep them separate from raw primary shares.
No same-format flag is available for original B3: its assigned-support sensitivity is NA.
The Q1-1-Z2 T2/T3 coarse-location flag remains. Q1-3-Z3 is excluded.

Actual data source: original automatic V90 on B3, frozen V90 transfer adapter on other
positions. Adapter approximations remain documented; independent temporal GT is unavailable.
The old three-series workbook used V35/V66 and is not relabelled as V90 here.

Class PNG codes: 0 background, 1 existing, 2 new, 255 baseline/unassigned.
The uncertainty-retained PNG also uses 255 for original flagged root pixels.
The raw group map preserves exactly the previous new-root pixels; the uncertainty
map intentionally withholds flagged pixels rather than asserting a class.
On-white maps use blue for existing, gold for new and gray for unassigned/baseline.

V90_two_class_rates.xlsx contains all66 observed frames,72 planned cells, field/tube
means and quality sensitivities. The Word file has editable results and a three-line table.
two_class_policy.json fixes the downstream definitions for later analyses.

Portable numeric reproduction (NumPy, Pillow):
python recompute_from_skeleton.py --bundle . --output /new/empty/reproduction
Cached native skeleton support, endpoint weights and original labels reproduce all66
fractions without model inference. The saved check matched every rate within1e-8.

To group a future V90 prediction, without merging identities:
python regroup_v90_prediction.py --classes /path/Tn_classes.png \
 --ownership /path/Tn_ownership.npz --output /new/empty/two-class-prediction
To rebuild raw extraction, calculate_two_class_rates.py needs the source project and
its active manifest; use --output with a new directory. To redraw figures, use
make_two_class_figures.py --fonts /path/to/licensed/Times-New-Roman/fonts.
Fonts are not redistributed. All diagrams use black Times New Roman text on white.
Python dependencies: numpy, Pillow, scikit-image, matplotlib, openpyxl, python-docx.
No training, annotation editing, image registration or original-file overwrite was performed.
'''
 (W/'README.txt').write_text(readme)
 # Render raw and uncertainty-retained categorical masks; keep native dimensions.
 for folder in sorted((W/'two_class_native').iterdir()):
  for p in list(folder.glob('*classes*.png')):
   if p.stem.endswith('_on_white'):continue
   m=np.array(Image.open(p));rgb=np.full((*m.shape,3),255,np.uint8);rgb[m==1]=(63,112,145);rgb[m==2]=(213,154,51);rgb[m==255]=(177,177,177)
   Image.fromarray(rgb).save(p.with_name(p.stem+'_on_white.png'))
 wb=Workbook();wb.remove(wb.active)
 for name,file in [('Series_adjacent','Series_mean_adjacent.csv'),('All_66_images','V90_two_class_all_66_frames.csv'),('All_72_cells','V90_two_class_all_72_planned_cells.csv'),('By_tube','Per_tube_two_class_means.csv'),('Gap_inclusive','Series_mean_previous_observed.csv'),('Tube_balanced','Series_mean_tube_balanced.csv'),('Exclude_coarse','Series_mean_excluding_coarse.csv'),('Assigned_support','Series_mean_assigned_support.csv')]:
  rr=read(W/file);ws=wb.create_sheet(name);keys=list(rr[0]);ws.append(keys)
  for r in rr:
   values=[]
   for k in keys:
    v=r[k]
    if v=='':v=None
    elif k not in ['material','position_id','tube_id','period','previous_observed_period']:
     try:v=float(v)if any(c in v for c in '.eE')else int(v)
     except(ValueError,TypeError):pass
    values.append(v)
   ws.append(values)
  for c in ws[1]:c.font=Font(name='Times New Roman',bold=True,color='000000');c.fill=PatternFill('solid',fgColor='E4EBF0')
  for row in ws.iter_rows(min_row=2):
   for c in row:c.font=Font(name='Times New Roman',size=10,color='000000')
  for j,k in enumerate(keys,1):
   ws.column_dimensions[ws.cell(1,j).column_letter].width=min(45,max(14,len(k)+2))
   if k.endswith('_pct')or k.endswith('_px')or'_pct_'in k:
    for row in ws.iter_rows(min_row=2,min_col=j,max_col=j):row[0].number_format='0.00'
  ws.freeze_panes='D2';ws.auto_filter.ref=ws.dimensions
 ws=wb.create_sheet('Definitions');ws.column_dimensions['A'].width=120
 for line in readme.splitlines():ws.append([line])
 wb.save(W/'V90_two_class_rates.xlsx')
 doc=Document();s=doc.sections[0];s.left_margin=Mm(15);s.right_margin=Mm(15);s.top_margin=Mm(16);s.bottom_margin=Mm(16)
 sty=doc.styles['Normal'];sty.font.name='Times New Roman';sty.font.size=Pt(10);sty.font.color.rgb=RGBColor(0,0,0);sty.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 doc.add_paragraph('V90-derived root composition: new and existing roots').runs[0].bold=True
 doc.add_picture(str(W/'V90_three_series_two_class_rates_SCI.png'),width=Mm(180));doc.add_paragraph(caption)
 doc.add_paragraph('Table 1. Predicted two-class skeleton-length shares (mean ± field SD).').runs[0].bold=True
 table=doc.add_table(rows=1,cols=5)
 headers=['Series','Period','Fields/tubes','New roots (%)','Existing roots (%)']
 for c,t in zip(table.rows[0].cells,headers):c.text=t
 for m in ['0353','A017','Q1']:
  for t in ['T2','T3','T4']:
   r=stats[m][t];values=[m,t,r['n_positions']+'/'+r['n_tubes'],f"{float(r['new_mean_pct']):.2f} ± {float(r['new_SD_pct']):.2f}",f"{float(r['existing_mean_pct']):.2f} ± {float(r['existing_SD_pct']):.2f}"]
   for cell,val in zip(table.add_row().cells,values):cell.text=val
 borders=OxmlElement('w:tblBorders')
 for side in ['top','bottom','left','right','insideH','insideV']:
  e=OxmlElement('w:'+side);e.set(qn('w:val'),'single'if side in ['top','bottom']else'nil');e.set(qn('w:sz'),'10');e.set(qn('w:color'),'000000');borders.append(e)
 table._tbl.tblPr.append(borders)
 for cell in table.rows[0].cells:
  tcPr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders');e=OxmlElement('w:bottom');e.set(qn('w:val'),'single');e.set(qn('w:sz'),'6');b.append(e);tcPr.append(b)
  for p in cell.paragraphs:
   for run in p.runs:run.bold=True
 doc.add_paragraph(cn);doc.add_paragraph(cn2);doc.add_paragraph(cn3);doc.add_paragraph(cn4)
 doc.add_paragraph(english);doc.save(W/'V90_two_class_rates_and_methods.docx')
 print('TWO_CLASS_TABLES_METHODS_AND_NATIVE_COLOURS_COMPLETE',flush=True)
if __name__=='__main__':main()
