# 根系时序配准结果（水珠排除版）

以 T1 为固定坐标系，RoMa 对 T2、T3 提取跨时期对应，再由 RANSAC 估计几何变换。原始图像没有修改；水珠只从变换拟合、质量指标和根系/土壤对比中排除，完整配准图仍保留原像素。

## 下载包

- [`registration_fullframe_core.zip`](registration_fullframe_core.zip)（约 75 MB）：脚本、中文说明、T1、T2/T3 的 T1 原尺寸配准图、透明 NoData 图层、排水珠根系/土壤叠加图、覆盖掩膜、变换和指标。
- [`registration_T2_visuals.zip`](registration_T2_visuals.zip)（约 60 MB）：T2 匹配点、混合叠加、棋盘格、候选根线变化、水珠审计、差异图和位置图。
- [`registration_T3_visuals.zip`](registration_T3_visuals.zip)（约 54 MB）：T3 对应可视化；未拍到的 T1 区域明确显示为 NoData。
- [`registration_full_extent_layers.zip`](registration_full_extent_layers.zip)（约 66 MB）：保留 T2/T3 全部原图范围的扩展画布 RGBA、排水珠全图拼接、RoMa warp、变换及覆盖掩膜。

四个 ZIP 均已通过 `unzip -t` 完整性检查。

脚本也可直接查看：[`scripts/register_root_timeseries.py`](scripts/register_root_timeseries.py)；运行说明见 [`scripts/ROOT_REGISTRATION_README.md`](scripts/ROOT_REGISTRATION_README.md)。汇总指标位于 [`metrics/`](metrics/)。

## 如何看图

- `registered_to_T1.png`：严格为 T1 原尺寸；只显示该时期真正落入 T1 视野的部分。
- `registered_to_T1_rgba.png`：同一结果，未观测区域透明。
- `root_soil_overlay.png`：T1 为洋红、后期为绿色、一致结构趋近白色；水珠或 NoData 为中性灰。
- `root_change_overlay.png`：启发式候选根线，T1 洋红、后期绿色、6 px 内重合为白色；它不是定量根分割。
- `full_extent_root_soil_overlay.png`：在扩展坐标画布上保留两期完整视野，用于查看全图拼接位置。
- `droplet_nuisance_mask.png` / `droplet_exclusion_overlay.png`：水珠排除范围的审计图。

## 当前图像的客观结果

- T2 → T1：T1 覆盖 96.98%，属于可信全幅配准；排水珠内点 3928/5000，RMSE 1.43 px。
- T3 → T1：T3 顶部对应 T1 底部，T1 覆盖 42.40%；排水珠内点 1857/2500，RMSE 1.61 px。
- 保守水珠掩膜覆盖 T1/T2/T3 约 5.44%/8.37%/2.90%。排除水珠后，根状线与土壤子集仍共同支持同一几何位置。

T3 剩余约 57.6% 位于 T1 拍摄边界下方，因此在 T1 原尺寸结果中必须为透明/NoData；完整 T3 已保留在扩展画布包中。强行把整张 T3 压入 T1 会制造不存在的像素对应，不适合根系生长追踪。

## SHA-256

```text
6de6a7b9bfdc2c6b5e4eef529b69b746a32a25574e2709aad121ed4326fc5ead  registration_fullframe_core.zip
ba89c226f729c52a6296d30f73600acad3ebc5592631dbb7be4285e6535914a1  registration_T2_visuals.zip
d88fcd3db4f8b687ca7bb0965632e7edb3d4e57e2cf374c598e16b72e943edfd  registration_T3_visuals.zip
e450be5fdbbb4a6c13aa5f1bf39dee3c041efdede3c9d057a545acc9b66087fd  registration_full_extent_layers.zip
```
