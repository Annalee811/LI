Table 1. Skeleton root length and temporal root-class length composition across three minirhizotron image series.

| Material | Period | nL / nI | Skeleton length (10³ px) | Persistent (%) | New (%) | Extension (%) | E / Lprev (%) |
|---|---|---|---|---|---|---|---|
| 0353 | T1 | 6 / — | 18.48 ± 12.44 | — | — | — | — |
| 0353 | T2 | 6 / 6 | 25.16 ± 17.55 | 56.71 ± 16.83 | 35.76 ± 15.90 | 7.53 ± 3.60 | 11.27 ± 10.01 |
| 0353 | T3 | 6 / 6 | 26.43 ± 19.15 | 82.40 ± 5.15 | 13.37 ± 3.96 | 4.22 ± 2.20 | 4.33 ± 2.23 |
| 0353 | T4 | 6 / 6 | 21.79 ± 18.07 | 86.89 ± 19.92 | 10.09 ± 17.18 | 3.02 ± 2.98 | 3.01 ± 4.28 |
| A017 | T1 | 6 / — | 13.06 ± 7.30 | — | — | — | — |
| A017 | T2 | 6 / 6 | 15.21 ± 7.54 | 51.39 ± 17.18 | 37.83 ± 16.55 | 10.79 ± 5.56 | 12.54 ± 4.72 |
| A017 | T3 | 4 / 4 | 13.55 ± 1.50 | 59.54 ± 18.45 | 25.95 ± 16.13 | 14.51 ± 2.71 | 17.12 ± 13.37 |
| A017 | T4 | 6 / 4 | 9.01 ± 4.26 | 76.65 ± 11.85 | 11.90 ± 9.77 | 11.45 ± 2.23 | 6.73 ± 2.78 |
| Q1 | T1 | 5 / — | 21.22 ± 21.83 | — | — | — | — |
| Q1 | T2 | 3 / 3 | 56.97 ± 81.11 | 52.32 ± 27.40 | 40.50 ± 23.29 | 7.18 ± 7.33 | 18.78 ± 22.29 |
| Q1 | T3 | 4 / 2 | 21.37 ± 26.15 | 59.87 ± 20.58 | 30.70 ± 19.43 | 9.43 ± 1.14 | 9.67 ± 4.80 |
| Q1 | T4 | 5 / 3 | 58.87 ± 60.74 | 82.39 ± 4.64 | 8.18 ± 4.67 | 9.44 ± 8.92 | 4.23 ± 2.65 |

Note 1. Values are means ± sample SD across available observation fields. nL indicates the number of available length observations; nI indicates the number of real adjacent-period pairs. The 18 observation fields belong to 8 source tubes and are not assumed to be independent biological replicates.
Note 2. Root length is measured in 10³ native registered pixels, with artificial stitching seams excluded. Physical calibration has not been confirmed; these values are not millimetres or whole-tube root-system lengths. Field area and sampling scale differ among sequences.
Note 3. For T2, T3 and T4, temporal quantities refer to T1–T2, T2–T3 and T3–T4, respectively. Persistent, new and extension shares are each current-class skeleton length divided by current total skeleton length ×100%. E/Lprev is current extension-class skeleton length divided by previous total skeleton length ×100%; it is an interval ratio, not a time-normalized elongation rate.
Note 4. Length means use all available observations in each period; temporal means use only valid adjacent pairs. These supports may differ. Missing observations and unavailable source files remain NA. Cross-gap comparisons are excluded. T1 is a baseline and has no three-class temporal percentages.
Note 5. All results are frozen-model estimates on matched fields: existing U-Net binary segmentation followed by the existing U-Net temporal head. Registration review flags are retained. No new training, physical conversion, independent accuracy claim or significance test is introduced. Rounding may cause displayed class percentages to differ slightly from 100%.

结果分析与讨论

本表汇总了0353、A017和Q1三组材料的18个观测序列，共63张可用图像及40组真实相邻时期配对。根系与背景分割提供各观测视野的根系掩膜，骨架测量获得总根长，时序分类进一步将可见根长划分为持续、新生和延伸三类。统计采用相同的配准裁剪视野，避免将完整扫描图的根长与局部时序组成混算。微根管研究强调对相同观测位置进行重复采集，并结合采样间隔解释根系动态[1]；因此，本研究同时列出各期根长观测数nL和有效相邻配对数nI。

0353的平均骨架根长由T1的18.48×10³ px增至T2的25.16×10³ px和T3的26.43×10³ px，随后在T4降至21.79×10³ px。其六个观测序列在四期均有可用结果，因而该组时期变化具有一致的观测支持。T1–T2的新生根长占比为35.76%，随后降至13.37%和10.09%；持续根长占比则由56.71%升至82.40%和86.89%。延伸根长占比依次为7.53%、4.22%和3.02%。这些模型估计共同表明，该组观测视野早期有较高的新增根长贡献，后期可见根系组成更多由既有根段构成。持续类占比上升本身并不等同于根系停止生长，也不能作为根系存活概率。

A017在三个相邻区间的延伸根长占比分别为10.79%、14.51%和11.45%，在本批可用观测中均高于0353相应区间的数值。其中T2–T3的延伸根长占比最高，为14.51±2.71%，以此前一期根长为分母的相对延伸比值为17.12±13.37%。但该区间有效配对仅4组，且该组T4全体可用视野的平均根长降至9.01×10³ px。因此，较高的延伸组成不代表总可见根长必然增加：新根或延伸根段的出现可以与其他根段的不可见、漏检或可见长度减少同时发生。根系分类和性状测量应采用清晰且一致的定义，避免把不同分母的百分比当作同一种生长指标[2]。

Q1在T1–T2和T2–T3的新生根长占比分别为40.50%和30.70%，至T3–T4降至8.18%，同时持续类占比升至82.39%。然而，Q1各期可用根长观测数为5、3、4和5，根长的离散程度也较大，例如T2为56.97±81.11×10³ px。不能将这种样本组成变化解释为同一批根系的剧烈生长波动。以T3–T4真正配对的3个观测序列为例，其平均根长实际由24.29×10³ px变为22.37×10³ px，而主表使用所有可用视野得到的T4平均值为58.87×10³ px。两者差异说明：判断相邻期根长变化必须使用相同位置的配对记录，不能直接相减不同样本组成的时期均值[1]。

本研究还区分了“延伸根长占比”和“相对延伸比值”。前者为E(t)/L(t)×100%，反映当期可见根长中延伸类的组成；后者为E(t)/L(t−1)×100%，反映延伸类根长相对于此前一期总根长的大小。二者均不同于净根长变化[L(t)−L(t−1)]/L(t−1)，也不同于按实际时间间隔计算的mm d⁻¹。这里的新生类表示模型判定的新近可见根段，尚不能直接替代经逐根跟踪确认的出生率。根长减少也不能单独证明根死亡或周转增加；微根管观测中的采样频率、位置稳定性及根段可见性都会影响动态估计[1]。

从方法学角度看，将分割结果转换为骨架长度，并进一步统计时序类别组成，可以同时呈现根系总量和组成的变化。RootPainter以校正性标注支持生物图像分割[3]，RootNav 2.0则结合分割与根系结构解析提取根系表型[4]。这些工作为本研究的自动化测量流程提供了方法学参照，但不替代对当前新观测视野的人工核验。后续应重点抽查细根、交叉处和低对比度根段，核对分类与测长结果是否一致，而不是只检查整图分割是否看起来相似。

上述结果应解释为观测视野层面的描述性模型估计。目前各序列裁剪面积及原始采样尺度可能不同，像素到物理长度的标定和实际采样日期也未确认，因此不能据像素根长直接判定整株根系规模、材料优劣或每日延伸速度。18个观测位置嵌套于8个源管号，不能作为18个独立微根管开展显著性检验；部分配准还保留待复核标记。补齐标定、时间间隔及位置复核后，宜报告经过面积或体积标准化的根长指标，并按源管号处理重复观测结构，以支持更可靠的生物学比较[1,2]。

参考文献

[1] Johnson MG, Tingey DT, Phillips DL, Storm MJ. Advancing fine root research with minirhizotrons. Environmental and Experimental Botany. 2001;45(3):263–289. https://doi.org/10.1016/S0098-8472(01)00077-6
[2] Freschet GT, Pagès L, Iversen CM, et al. A starting guide to root ecology: strengthening ecological concepts and standardising root classification, sampling, processing and trait measurements. New Phytologist. 2021;232(3):973–1122. https://doi.org/10.1111/nph.17572
[3] Smith AG, Han E, Petersen J, et al. RootPainter: deep learning segmentation of biological images with corrective annotation. New Phytologist. 2022;236(2):774–791. https://doi.org/10.1111/nph.18387
[4] Yasrab R, Atkinson JA, Wells DM, French AP, Pridmore TP, Pound MP. RootNav 2.0: Deep learning for automatic navigation of complex plant root architectures. GigaScience. 2019;8(11):giz123. https://doi.org/10.1093/gigascience/giz123
