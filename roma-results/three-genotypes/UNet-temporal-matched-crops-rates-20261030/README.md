# U-Net时序分割与相邻时期根长比例

完成63张可用图像的类别输出，沿用上一批二元掩膜。所有时序类别图来自冻结模型真实推理；不重新训练。

| 种质 | 区间 | 有效n | 持续根占比(%) | 新生根占比(%) | 延伸根占比(%) |
|---|---|---:|---:|---:|---:|
| 0353 | T1-T2 | 6 | 56.7100 | 35.7605 | 7.5295 |
| 0353 | T2-T3 | 6 | 82.4047 | 13.3714 | 4.2239 |
| 0353 | T3-T4 | 6 | 86.8859 | 10.0906 | 3.0235 |
| A017 | T1-T2 | 6 | 51.3856 | 37.8271 | 10.7873 |
| A017 | T2-T3 | 4 | 59.5418 | 25.9492 | 14.5090 |
| A017 | T3-T4 | 4 | 76.6543 | 11.9001 | 11.4456 |
| Q1 | T1-T2 | 3 | 52.3223 | 40.4972 | 7.1805 |
| Q1 | T2-T3 | 2 | 59.8676 | 30.6998 | 9.4326 |
| Q1 | T3-T4 | 3 | 82.3882 | 8.1763 | 9.4354 |

主表三类比例的分母是当前期总骨架根长，合计100%；Excel另列以前一期总根长为分母的比值。先逐重复计算，再对有效重复求均值。

## 方法与解释

Models: Frozen V35 U-Net epoch77 supplies root probabilities; frozen V66 U-Net temporal head epoch37 supplies temporal labels. No retraining or added ConvLSTM/GAT.

Foreground: The previous batch of63 matched-crop U-Net masks is immutable. Temporal class pixels union exactly equals each existing root mask; root widths and lengths are unchanged.

Current-length share: Each temporal class skeleton length/current total skeleton root length ×100%; persistent+new+extension sum100%. These are length compositions, not daily rates.

Previous-length ratio: Each current temporal class skeleton length/preceding-frame total skeleton root length ×100%. The three ratios need not sum100%; no time-duration normalization is implied.

Skeleton lengths: Unique8-neighbor skeleton edges have axial length1 and diagonal length√2, suppressing duplicate diagonal edges with an orthogonal bridge. At a class boundary each edge is divided equally between its two endpoint classes. Isolated skeleton pixels count1.

Adjacent means: Compute each available replicate first, then take unweighted arithmetic means over real T1-T2,T2-T3,T3-T4 pairs. Report actual n and sampleSD; missing observations/sources are NA, not0.

Missing dates: For display only, a frame after a missing acquisition is classified relative to the preceding actually observed frame, with its gap recorded. These longer intervals are excluded from adjacent means. A T4-only sequence remains unclassified.

Classes: 0 background;1 persistent;2 new;3 extension;4 T1baseline;5 root without prior observation. Blue baseline and gray unclassified roots are not temporal classes.

Experimental units: Six sequence identifiers per genotype are retained with source tube identifiers. Outputs are descriptive; independent biological-replicate mapping and significant differences are not inferred from folder names.

Scope: Registered matched cropped fields on their frozen native pixel grids; seams are excluded. Length units are pixels. No millimeter/day conversion without calibration and acquisition dates.

Registration: Existing review_required flags are retained. Flagged fields remain in all-available descriptive means and are counted explicitly. The Q1-1-Z2 T2-T3 registration is coarse/unverified.

Accuracy: These are frozen-model estimates on new fields, without new manual temporal ground truth; no new Dice or accuracy is asserted.

