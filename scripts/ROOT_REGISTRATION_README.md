# 根系时序图像全局配准

脚本：`scripts/register_root_timeseries.py`

## 运行

当前仓库已经配置好 Python、PyTorch 与 RoMa，不需要重新安装依赖：

```bash
cd /workspace/LI
.venv/bin/python scripts/register_root_timeseries.py \
  --data-dir data \
  --output-dir results/registration/full_frame
```

脚本支持两种输入结构：

```text
data/T1.png, data/T2.png, data/T3.png, data/T4.png
```

或批量目录：

```text
data/T1/sample.png
data/T2/sample.png
data/T3/sample.png
data/T4/sample.png
```

目录方式要求不同时期使用相同文件名。缺少尚未拍摄的 T4 时，脚本会处理已有时期。

## 方法

1. RoMa 提取跨时期稠密对应和置信度。
2. 用严格圆周径向梯度验证检测水珠，并用多方向长线结构保护细根；匹配点任一端落入水珠掩膜时即排除。
3. 根系/线状高对比区域在取样时提高权重，同时保留非水珠土壤纹理以稳定全局相机变换。
4. RANSAC 分别估计 similarity、affine 和 homography 候选。
5. 对两幅图都检查重合率与匹配的网格覆盖。全幅支持充足时使用 homography；只有局部真实重合时使用外推更稳定的 similarity。
6. 所有源图整体变换一次，不修改原图，也不把低置信度 RoMa 稠密场强行外推到未拍摄区域。

默认启用 `--ignore-droplets`。水珠只在变换估计、质量统计和对比图中被排除；`registered_to_T1.png` 和扩展全图仍保留原始像素。可用 `--no-ignore-droplets` 关闭，或用 `--droplet-hough-threshold` 调整检测保守程度（值越大，掩膜越小）。

## 主要输出

每个时期目录都包含：

- `registered_to_T1.png`：严格为 T1 原尺寸，只有真实拍到并落入 T1 的像素；
- `registered_to_T1_rgba.png`：同一结果，NoData 区域透明；
- `registered_on_T1_context.png`：在变暗的完整 T1 背景上显示该时期的真实位置；
- `full_extent_registered.png`：保留该时期完整原图的扩展坐标结果；
- `full_extent_overlay.png`：T1 与该时期完整视野的扩展画布拼接；
- `root_soil_overlay.png`、`overlay_blend.png`、`checkerboard.png`：排除水珠后的配准对比，水珠/NoData 显示为中性灰；
- `full_extent_root_soil_overlay.png`：保留两期完整视野且将水珠降权的扩展拼接图；
- `root_change_overlay.png`：保守的候选根线变化辅助图，仅用于观察，不是根系分割定量结果；
- `droplet_nuisance_mask.png`、`droplet_exclusion_overlay.png`：水珠掩膜及审计图；
- `matches.png`：RoMa 匹配点；
- `coverage_mask.png`、`placement_diagram.png`：真实覆盖范围；
- `transform.json`、`roma_warp.npz`、`metrics.json`：变换、warp 与质量指标。

序列目录还包含：

- `temporal_RGB_T1_T2_T3.png`：T1=红、T2=绿、T3=蓝，重合结构趋近白色；
- `temporal_overview_preview.png`：T1、T2、T3 在 T1 坐标中的并排预览；
- `summary.csv` 和 `summary.json`：批量指标汇总。

当同一点位有两个或更多时期时，还会生成 `T1/common_overlap/`：

- `T1_aligned_crop.png`、`T2_aligned_crop.png` 等：所有时期完全相同的 T1 坐标、尺寸与像素尺度；
- `aligned_sequence_montage.png`：同位置原始裁剪的全分辨率并排拼接；
- `root_soil_sequence_montage.png`：统一排除各时期水珠后的并排图；
- `overlay_T1_T2.png` 等：T1 与每个时期的两两叠加；
- `pairwise_root_soil_overlays.png`：所有两两叠加的横向拼接；
- `checkerboard_T1_T2.png` 等：同位置棋盘格检查；
- `common_crop.json`：共同裁剪框、可比较面积及处理策略。

共同裁剪是所有时期真实覆盖区域内的最大轴对齐矩形；区域外不参与拼接。当前阶段只做变化可视化，不自动判定根系新增、加长或消失。

## 当前三幅图的客观覆盖

- T2→T1：T1 覆盖约 97.0%，属于可信全幅配准；排水珠匹配 RMSE 约 1.43 px。
- T3→T1：T3 顶部对应 T1 底部，T1 覆盖约 42.4%；排水珠匹配 RMSE 约 1.61 px。T3 其余约 57.6% 投影在 T1 下方，不在 T1 的拍摄视野内。

当前保守水珠掩膜覆盖 T1/T2/T3 约 5.44%/8.37%/2.90%。排除水珠后，T2、T3 的位置解与独立的根状线和土壤子集验证一致，因此 T3 的部分重合结论不是由水珠造成。`metrics.json` 还报告低通 Lab 土壤通道的 trimmed-NCC；该指标只评价真实重合带，不能把 T3 的 42.4% 重合解释成全图覆盖。

因此 T3 的 T1 原尺寸图上部必须是透明/NoData。把完整 T3 压缩到 T1 原尺寸会破坏已经由 RoMa、相关搜索、SIFT/ORB 和 ECC 共同验证的真实位置，并制造假的根系迁移，不能用于根系变化定量。

如需程序在发现部分重合时直接报错，可加：

```bash
--strict-full-overlap
```

若已经有此前生成的 RoMa warp，可只重建后处理结果：

```bash
.venv/bin/python scripts/register_root_timeseries.py \
  --data-dir data \
  --output-dir results/registration/full_frame \
  --reuse-warp-dir results/registration
```
