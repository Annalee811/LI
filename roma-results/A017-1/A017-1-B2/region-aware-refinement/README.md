# B2 最新结果：同一观察区域配准，保留季节变化

T1为苗期，实际序列T1→T2→T4，T3缺失。定位同一区域，允许新生根、根段延伸、局部位移与土壤变化。不能要求各时期每条根与土壤纹理完全相同。

全原图、多尺度结构搜索结合用户提示的横根与分叉组合确认目标条带。RGB仅采用整体等比缩放、旋转、平移，原色从原图采样一次，不应用RoMa非刚性场。RoMa在更大的相邻共同视野里重新运行上下文窗口，不限制对应点必须与全局位置相差≤3像素；置信度保持实际模型输出。

- [共同区域裁剪原图拼接](common_original_RGB_montage.png)
- [根系细节观察](root_region_detail_comparison.png)
- [原图上的区域定位框](region_localization_original_canvases.png)
- [六类综合结果](six_results_region_comprehensive.png)
- [区域与局部位移证据](region_and_local_change_evidence.png)
- [T1→T2 更大视野六类结果](T1_to_T2_six_results_full_view.png)
- [稳定结构匹配锚点（细线）](T1-T2_stable_region_anchor_comparison.png)

## 可下载结果

- [共同连续区域的RGB、六类图、细节、评估、脚本](A017-1-B2_region_results.zip)
- [T1→T2 更大共同视野独立图像及六类结果](A017-1-B2_region_T1-T2_images.zip)
- [T2→T4 独立图像及六类结果](A017-1-B2_region_T2-T4_images.zip)
- [原始像素裁剪和完整T1画布RGB](A017-1-B2_region_original_RGB.zip)
- [整体区域定位坐标场](A017-1-B2_region_global_fields.zip)
- [T1→T2 RoMa候选稠密场与置信度](A017-1-B2_region_T1-T2_RoMa_fields.zip)
- [T2→T4 RoMa候选稠密场与置信度](A017-1-B2_region_T2-T4_RoMa_fields.zip)

共同裁剪的独立无标注原色PNG：[T1](T1_common_original_RGB.png)、[T2](T2_common_original_RGB.png)、[T4](T4_common_original_RGB.png)。原始像素裁剪也单独保留，没有插值；其原始尺寸不同，不能混作已配准画布。

[评估摘要](result_summary.md)／[精确指标](evaluation.json)。共同可观察区域是几何可见范围；可靠点对应掩膜只是模型证据范围，不用它裁掉新生根与季节变化。局部位移箭头不是生长方向，不能据此自动认定土层迁移、新生或死亡。

区域定位规则与旧版严格同位置逐点规则分别保存，不能混用通过率。自动误差来自描述符共识锚点内空间分块验证；没有独立人工对应点，真实TRE未评估。T2原宽1330像素，统一尺度和600 DPI不会新增未采集的细节。完整Python代码、初始区域定位资料和数值场保留，旧版不覆盖。
