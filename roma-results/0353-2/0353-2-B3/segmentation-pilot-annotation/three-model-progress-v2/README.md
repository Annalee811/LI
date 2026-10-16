# 0353-2-B3 三模型选优与四期预测

先阅读“结果说明.txt”。根分割比较包括U-Net、上传的TensorFlow DeepLabV3+、新训练的SegFormer-B0。由于本次T1/T2标注逐像素等同上一轮，前两模型复用实际完成的训练；新增SegFormer67轮试训，按验证Dice选择U-Net。

时序模块是选中U-Net架构的独立8通道模型：前一期RGB、当前期RGB、冻结分割模型的前后期根概率。训练只使用T1→T2。此模块并非三种架构时序类别排名。T3/T4没有完整复核标注，四期颜色图不是已验证的根系追踪真值。延伸根验证Dice为0，当前不可用于可靠识别。

两份压缩包使用同一顶层目录，可以解压到同一位置；分别打开图片也无需合并。

训练包保留：100个原始256×256图块、原人工掩膜与属性、分组清单、选中分割权重、SegFormer权重、时序权重、评估、完整脚本。相同空间位置T1/T2同组，组间64像素隔离。仅一个位置、两次已复核采集，不代表独立位置泛化。

预测包保留：四期上部原色RGB、根掩膜、根概率16位PNG、原始时序类别、含未判定版本、时序置信度、白底根系图、RGB叠加图及综合图。类别编码见four_period_prediction_summary.json。灰色为未判定；持续表示相对上一期，并不表示全部来自T1。未推断死亡或单根ID。

original_manual_label_review直接解码用户上传的原始标注，与模型预测分开。T1蓝色基础根；T2紫色持续根69201像素、绿色新生根37360像素、黄色延伸根5824像素（上部区域）。包含原标注RGB叠加、白底标签、原始掩膜及属性。三类原标签全部保留并进入时序模型；延伸验证Dice为0是模型失败，不是缺少该类标注。

editable_T1T2_manual_T3T4_model.json可导入自选图片版annotate.html：先选择原来四张完整2541×1867的RGB，再导入。原T1/T2完全保留，T3/T4全部按模型预测，原有笔迹完全忽略、不合并；非预测根位置仍为unknown64，预测未复核。原始输入进度单独备份为original_human_progress.json。

复现环境：torch==2.8.0，numpy==2.2.6，Pillow==11.3.0，matplotlib==3.10.6，transformers==4.57.1。自动检测CUDA；本次CPU，4线程。

合并解压后复现四期预测：
```
python source/predict_four_temporal.py --run . --dataset data_run --originals four_period_predictions --upper-only
```

重训第三模型：
```
python source/train_segformer_progress.py --dataset data_run --output segformer_retrained
```

重训时序模块：
```
python source/train_pair_temporal.py --dataset data_run --selection three_model_selection.json --segformer segformer --output temporal_retrained
```

上一轮实际训练的原始TensorFlow DeepLabV3+大权重和代码仍可下载：
https://raw.githubusercontent.com/Annalee811/LI/62a4bc9979fdb823fa159eaef503a2ade7411f38/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/trained-upper-t1t2-v1/0353-2-B3_DeepLabV3plus_TF112_model.zip

训练中的两次早期损失探索仅依据验证集调整，均曾偏向持续根。最终条件分类损失只作用于人工根像素，使用类别平衡权重及按组预测的两期根概率；当前为探索性预实验，而非冻结方案的独立验证研究。
