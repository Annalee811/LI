# 全系列原图、独立拼接RGB与定位质量（Q1已更正）

[下载单个完整ZIP（约1.99GB）](https://github.com/Annalee811/LI/raw/9ec7dc3c4dd5b58cad58fd1ea9b0aea4e7138edb/roma-results/all-series-localization-v2/all_series_original_and_stitched_RGB_v2.zip)

包含各期原图、66张独立拼接RGB、全系列与各位置定位总览、共同面积统计、累计置信度图和原生float32数值、观察区域与对应质控掩膜，以及完整Python脚本。0353-2-B3仅输出此前标注使用的上半观察区，四期RGB均2541×1108。其余原图保留原始文件字节。

[逐位置汇总](all_series_position_summary.csv) · [各期原尺寸及面积比例](all_series_common_area_ratios.csv) · [累计置信度及有效支持统计](all_series_cumulative_confidence_summary.csv) · [完整脚本](../../analysis-scripts/all_series_localization_v2/README.md)

面积以共同区域映射回该期原图后的多边形并集除以完整原图W×H，考虑尺度差异、重复区域及接缝。RGB不做弹性变形或颜色调整。置信度沿相邻实际观测时期的RoMa对应链累计，以已选定位为条件；没有人工标志点，不将覆盖率称为独立测量的准确率。

当前数据19个位置、69张输入图；18个位置有定位结果，其中8个位置含待复核观察段。Q1完全采用新版Q1.zip（SHA256:de74de5504474af5ef4f6e1cc28265a87ac7039032593e465ff9128336d394e2）。Q1-1-Z3为原先误置的三期；Q1-2-B3改用正确四期。Q1-3-Z3未建立连通T1的定位，整序列共同面积为未确定，另存T2/T3局部结果。旧Q1-1-B3不属于当前数据。

旧版ZIP保留作历史记录，当前RGB见RGB/目录，旧独立RGB入口也已替换。包内file_manifest.json及外部archive_manifest.json记录完整性；归档仅省去重复预览及辅助矩阵副本，累计C的float32值逐位一致。
