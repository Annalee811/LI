# 18个观测序列的根长与时序性状三线表

本次严格使用用户指定的冻结Excel明细，未重新训练、推理、测长或改变原始数值。原文的材料均值表已展开为18个观测编号各自的T1—T4记录，共72个时期单元。

保留用户指定的五项指标：Skeleton length (10³ px)、Persistent (%)、New (%)、Extension (%)、E / Lprev (%)。除定位所需的编号和时期外，展示表未增加其他指标列。

## 文件

- 18_series_individual_three_line_tables.docx：18张独立Word三线表，每个编号一张、T1—T4四行。PDF预览为6页，每页3表。
- 18_series_combined_three_line_tables.docx：合并总表版，按0353、A017、Q1分为3张三线表；每张列6个编号、24个时期记录，共72行。PDF预览为3页。
- 18_series_root_traits_tables.xlsx：可编辑Excel，含72行总表、18个编号分表、未舍入数值及定义。
- 18_series_values_full_precision.csv / extracted_values_full_precision.json：未舍入数值和原有数据状态。
- table_manifest.json / verification.json：来源、单位、缺失记录及逐单元核对结果。
- sources/：用户原链接的Excel、Word及下载SHA256记录，保留不变。
- make_tables.py / verify_tables.py：提取、制表与验证代码。

两版Word均为真正可编辑的表格，保留顶线、表头下线和底线，无竖线和表体网格线。正文及数字为黑色Times New Roman。展示值统一保留两位小数，Excel的Full_precision_72及CSV/JSON保留未舍入值。

## 数据口径

来源：Annalee811/LI，固定提交8b4fdb81b5938720e263d802426745c982c4479c，文件All_microtube_root_traits.xlsx，All_72_cells工作表。SHA256为ea74bf1853766e51fa3b30546d6893032e57e50ea991514c2735182a6fdb0e52。链接记录在sources/download_verification.json。

63条根长结果和40组有效相邻时期配对完全沿用该文件。6个时期未采集、3条源结果不可用，保持缺失；即使其他历史或当前目录中存在图像，也未补算该冻结表格的缺失值。

长度仅由原始Root_length_px除以1000，单位为10³ px，未换算为毫米。三个类别百分比直接复制各编号原有的当期长度组成。E / Lprev是延伸类根长相对此前一期总根长的比值，不是按天计算的延伸速度。T1为基线，不填时序类别比例。缺少真实相邻时期时，对应比例保留“—”；有效的零值仍显示0.00。未将跨缺失期的比较替代真实相邻配对。

这些是单个观测序列的模型估计，不是材料组均值±标准差。18个观测位置嵌套于8个源管号，未将其改称18个独立生物学重复。源表的配准复核标记保留为†；Q1-1-Z2的T2—T3粗匹配仍未验证。该来源使用既有V35二值与V66时序U-Net结果，不替换为V90结果。

## 复现

使用requirements.txt中的依赖，并在本地准备有许可的Times New Roman常规及粗体字体，文件名为Times New Roman.ttf和Times New Roman Bold.ttf；字体文件未随包分发。

```bash
python make_tables.py --fonts-dir /path/to/fonts
python verify_tables.py
```

PDF是使用同一组提取值另行排版的预览文件；Word内的表格可继续编辑。验证包括来源SHA256、指标定义一致性、两版Word共720个指标单元、三线表边框结构、Excel未舍入值、PDF页数和文字边界。
