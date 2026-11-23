# V90根段构成：新生根与既有根

后续V90性状占比统一采用两类：新生根沿用原类别2；既有根合并原持续类1与延伸类3。这是离散预测标签分组，不重新训练网络，不合并根轴实例，不重新选阈值。66幅图像的新生类与上一版新生专版逐像素一致，原根轴/track身份文件逐字节保留。

平均率按本期可见预测骨架长度计算。既有率为原持续长度与延伸长度之和除以本期总长，两类合计100%。既有指根轴身份，包括原根的新延伸部分，不能解释为每个像素前期均已存在。T1不赋比例，缺失时期不补0。

主图沿用视野等权的描述性平均口径，仅用真实相邻时期，共43个区间；5个跨缺失时期区间单列。均值和样本SD均在每幅影像先合并类别后计算，未相加两类SD。18个观察位置属于8个微根管，不当作18个独立生物重复。另附微根管平衡、全部48个前后观测区间、粗定位排除及已分配支持的敏感性表。

三个系列的新生根预测长度占比均在T2最高，既有根占比均在T4最高。全部数值来自既有V90预测；旧V35/V66平均率未改名充作V90。跨微根管真值不足，这些是模型预测构成，不是已验证准确率。

原不确定标记保留，主表原始暂定预测比例与排除不确定区域后的已分配比例分开给出。两种分母不能混用。背景0、既有1、新生2、未赋值255；255用于基准期或不确定状态，不是第三种生物根类。

## 下载

- [三系列两类平均骨架长度占比：折线图 JPG](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_rates_SCI.jpg)（0.88 MB）
- [三系列两层三维图 JPG](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_3D_SCI.jpg)（1.06 MB）
- [全部18位置数据、三系列均值与敏感性统计 Excel](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_two_class_rates.xlsx)（0.07 MB）
- [完整包：图表、全部数据、两类标签、原ID及可重算源码](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_two_class_rates_complete.zip)（30.03 MB）
- [三线表、方法与结果 Word](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_two_class_rates_and_methods.docx)（0.38 MB）
- [数据与标签完整性检查](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/Final_verification.json)（0.00 MB）
- [完整说明](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/README.txt)（0.00 MB）
- [方法与结果 中文文本](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/Two_class_methods_and_results_CN.txt)（0.00 MB）
- [三维图 矢量PDF](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_3D_SCI.pdf)（0.05 MB）
- [三维图 可编辑SVG](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_3D_SCI.svg)（0.04 MB）
- [三维图 600dpi TIFF](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_3D_SCI.tif)（1.08 MB）
- [图稿预览：V90_three_series_two_class_3D_SCI](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_3D_SCI_preview.png)（0.13 MB）
- [折线图 矢量PDF](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_rates_SCI.pdf)（0.05 MB）
- [折线图 可编辑SVG](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_rates_SCI.svg)（0.03 MB）
- [折线图 600dpi TIFF](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_rates_SCI.tif)（0.80 MB）
- [图稿预览：V90_three_series_two_class_rates_SCI](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/V90_three_series_two_class_rates_SCI_preview.png)（0.09 MB）
- [后续两类分组配置](https://raw.githubusercontent.com/Annalee811/LI/a4657f9454b07f48fdf376c3fb503b83b2a5b259/two_class_policy.json)（0.00 MB）

## 后续使用与复现

完整包中的two_class_policy.json固定分组定义。项目代码目录为`/workspace/LI/analysis-scripts/V90_two_class_traits_20261123`；本次独立输出为`/workspace/work/V90_two_class_rates_20261123`。原权重、标注、图像和旧实验保持原样。

未来预测可用regroup_v90_prediction.py分组；原身份文件直接复制，类别合并不合并不同根轴。随包recompute_from_skeleton.py可仅依赖NumPy/Pillow从保存的骨架端点权重重算全部66幅的两类比例，已与表格核对至1e-8。图稿采用Times New Roman黑字、白底、600dpi，并提供PDF/SVG。
