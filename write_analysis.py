from pathlib import Path
import json,zipfile,hashlib
from docx import Document
from docx.shared import Pt,Mm,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
W=Path(__file__).resolve().parent
TITLE='Controlled evaluation of temporal modules and structural decoding for root classification'
CN='''在冻结上游分割特征与V66时序先验、统一训练预算和验证集选权重的条件下，完整新增模块配置在B3内部留出区域的宏平均Dice为63.71±0.35%，较冻结V66参照的62.56%提高1.16个百分点（图A）。持续段、新生段和延伸段的Dice分别为97.28%、81.63%和12.23%，表明类别表现仍明显不均衡（图B）。配对模块消融显示，保留GAT、ConvLSTM和学习式身份门控分别对应0.76±0.13、0.30±0.47和0.02±0.11个百分点的宏平均Dice增益，其中GAT在三个随机种子下均表现为正向收益，而门控效应接近零，尚未显示稳定增益（图C）。在另行开展的完整视野配对结构解码实验中，各阶段使用同一检查点的相同神经输出，宏平均Dice由63.70±0.34%提高至根轴约束后的73.51±0.50%，并在加入连接处约束后达到76.71±1.99%（图D）；主要改善来自延伸段，其Dice由12.38±1.18%增至45.71±5.45%，新生段由81.48±0.17%提高至87.09±0.59%，持续段则保持在97%左右（图E）。然而，最终实例成对精确率、召回率及F1分别为95.91±0.45%、49.77±1.07%和65.52±0.85%，与仅加入根轴约束时接近，提示类别一致性的改善尚未转化为根身份完整恢复，根轴过分裂仍是主要限制（图F）。上述结果来自B3内部条件实验，三个随机种子的标准差仅反映优化波动；当前帧分类器在本轮预算下退化为持续类预测，不能代表充分调优的单帧模型。跨微根管预测尚缺人工时序真值，因此目前不能据此推断独立泛化准确率。'''
EN='''With frozen upstream segmentation features and V66 temporal priors, a matched optimization budget, and validation-based checkpoint selection, the full newly trained module configuration achieved an internal macro Dice of 63.71 ± 0.35%, compared with 62.56% for the frozen V66 reference (Fig. A). Dice scores were 97.28%, 81.63%, and 12.23% for persistent, newly observed, and extending roots, respectively, revealing substantial class imbalance in performance (Fig. B). Seed-matched differences between the full and ablated configurations were 0.76 ± 0.13, 0.30 ± 0.47, and 0.02 ± 0.11 percentage points for GAT, ConvLSTM, and learned identity gates, respectively (Fig. C). GAT produced a positive difference in all three optimization seeds, whereas the identity-gate effect was small and inconsistent. In a separate paired whole-field decoding experiment using identical neural outputs within each seed, macro Dice increased from 63.70 ± 0.34% for the raw head to 73.51 ± 0.50% with axis constraints and 76.71 ± 1.99% with axis and junction constraints (Fig. D). The largest class-specific change occurred for extending roots, whose Dice increased from 12.38 ± 1.18% to 45.71 ± 5.45%; newly observed roots increased from 81.48 ± 0.17% to 87.09 ± 0.59%, while persistent-root Dice remained approximately 97% (Fig. E). Final pairwise instance precision, recall, and F1 were 95.91 ± 0.45%, 49.77 ± 1.07%, and 65.52 ± 0.85%, respectively, and remained close to the axis-only values (Fig. F). Thus, improved temporal class agreement did not establish improved recovery of complete root identities, and fragmentation remained substantial. These results describe conditional internal B3 experiments; optimization-seed variation does not constitute biological replication. The current-frame classifier collapsed to persistent-only predictions under this budget and should be interpreted as a negative control rather than a fully tuned single-frame benchmark. Independent cross-tube accuracy remains unmeasured because human temporal ground truth is unavailable.'''
CAP='''Figure X. Controlled evaluation of neural modules and paired structural decoding for temporal root classification. (A) Internal macro Dice of a frozen V66 reference and five newly trained configurations, evaluated on common eligible root pixels. Each trained configuration used three fixed optimization seeds, 100 full-graph updates, and 8,192 sampled training root pixels per update. The historical V66 reference was not retrained. (B) Mean class Dice for persistent, newly observed (New), and extending roots. Cell shading spans 0–100%; printed values are means. (C) Seed-matched differences in macro Dice between the full configuration and each module ablation (full minus ablated; pp, percentage points). Positive values favor retaining the specified module. Removing ConvLSTM retains other temporal features and graph edges; removing learned identity gates retains GAT attention and the fixed geometric prior. (D) Macro Dice at successive structural decoding stages using the same whole-field neural outputs within each seed. Thin gray lines connect matched seeds. (E) Corresponding class-specific Dice. (F) Pairwise instance precision, recall, and F1 after axis-only or axis-plus-junction decoding, evaluated on 69,519 eligible pixels with reference root identities. These instance metrics differ from graph-edge identity-link F1. Points and error bars represent means and sample standard deviations over optimization seeds (n = 3), not confidence intervals or biological replicates. Small black points in A and C denote individual seeds; the frozen V66 reference is a single observation. Internal temporal class scores pool T2–T4 eligible typed root pixels (83,478 pixels: 69,879 persistent, 12,571 newly observed, and 1,028 extending), exclude background, and condition on a fixed foreground. The two rows use distinct inference contexts and must not be combined into one protocol. Upstream features and priors had historical B3 exposure; results are internal reference agreement rather than independent generalization estimates. Predictions are available for three additional tubes (seven positions, 26 frames), but no independent Dice or IoU is reported because human temporal ground truth is unavailable.'''
README='''SCI figure package — 2026-11-23

Six-panel figure, final size 190 x 145 mm; white background, black Times New Roman
text; 8-pt axis labels, 12-pt panel letters. JPG, PNG and lossless LZW TIFF are 600 dpi.
PDF embeds Times New Roman; SVG retains editable text and requires the font locally.
Use the vector PDF or TIFF for submission, subject to the target journal's size rules.

Source results are copied, hashed and included. No new training, inference,
registration or modification of old results was performed for this visualization.
The current figure reports the newly added conditional control experiments. Its
"Full" configuration is not a relabelling of historical V90 headline scores.

TOP ROW: internal partition-isolated module study.
BOTTOM ROW: separate paired whole-field structural-decoder study.
Do not splice the two contexts into a single before/after comparison.
Errors are sample SD over three optimization seeds, not confidence intervals or
biological replication. No statistical significance or P-values are implied.
Frozen V66 is one historical reference; current-frame-only is a collapsed negative
control under this budget. Identity-gate improvement is small and inconsistent.
Final root-identity recall remains low; cross-tube accuracy is unavailable without
human temporal ground truth. Empty annotation templates are not truth.

Results_and_figure_caption.docx contains editable English/Chinese results prose
and an English figure caption. Chinese_analysis.txt and English_results_and_caption.txt
provide plain-text copies; no tables are needed.

Rebuild:
python plot_sci_results.py --results /path/to/archived/reports \
  --fonts /path/to/licensed/Times-New-Roman/fonts --output /new/output
The plot uses only existing CSV/JSON files. Source copies have a source_ prefix;
to rebuild from this bundle alone, copy them to another results directory after
removing the source_ filename prefix. Fonts are not redistributed.
Python dependencies: numpy, matplotlib, Pillow; python-docx is needed only for
write_analysis.py. Sources and per-panel values are in this archive.
'''
def main():
 (W/'Chinese_analysis.txt').write_text(CN+'\n',encoding='utf-8')
 (W/'English_results_and_caption.txt').write_text(TITLE+'\n\n'+EN+'\n\n'+CAP+'\n',encoding='utf-8')
 (W/'README.txt').write_text(README)
 doc=Document();sec=doc.sections[0];sec.page_width=Mm(210);sec.page_height=Mm(297)
 sec.top_margin=Mm(18);sec.bottom_margin=Mm(18);sec.left_margin=Mm(15);sec.right_margin=Mm(15)
 style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(10);style.font.color.rgb=RGBColor(0,0,0)
 style.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');style.paragraph_format.space_after=Pt(7)
 doc.add_paragraph(TITLE).runs[0].bold=True
 doc.add_picture(str(W/'V90_controlled_experiments_SCI.png'),width=Mm(180))
 for text in [CAP,'Results',EN,'结果分析',CN]:
  p=doc.add_paragraph(text)
  if text in ['Results','结果分析']:p.runs[0].bold=True
 doc.save(W/'Results_and_figure_caption.docx')
 print('EDITABLE_ANALYSIS_AND_CAPTION_SAVED')
if __name__=='__main__':main()
