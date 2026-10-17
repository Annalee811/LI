# 三模型细根与连续性优化：复现说明

将本轮 ZIP 解压合并；每个 ZIP 都使用同一个 `0353-2-B3_three_model_v9` 根目录。模型结果分为三个 ZIP、框架权重分为两个 ZIP，以满足 GitHub 单文件小于100 MiB的限制。训练切块、原始标注和最终预测彼此独立保存。此处提供复现方法，正式结论见 `Results_CN.md` 与全部实测指标。

Python 3.12 使用 `source/paper_v6/requirements.txt` 中的版本。PyTorch 自动检测 CUDA，本轮实测为 CPU。原上传 DeepLabV3+ 必须用原始 TensorFlow 1.12/Python 3.5 环境；不是 torchvision 的替代模型。其 CPU Docker 镜像由官方基础镜像摘要与 SHA256 固定的官方 Intel TensorFlow wheel 构建，脚本 `runtime/prepare_deeplab_runtime.py` 保持 TLS 和校验和验证。已配好的环境可复用 `b3-tf112-mkl:local`。Docker 保持平台代理和 CA；不要关闭 TLS 或改用 host 网络。

所有命令在解压后的根目录执行。原始完整上半部 RGB、二值标注及原属性层位于 `data` 包，原始进度 JSON 保持字节不变。上半部未画笔迹的像素按此前已确认的完成声明视作背景；没有修改人工根像素来提高指标。细根的训练权重基于笔迹局部宽度，不能据此认定真实根径。

## 直接预测，不读取人工标注

```bash
python source/paper_v6/predict_three_v9.py --model unet --bundle . --rgb-dir INPUT_RGB --output predictions_unet
python source/paper_v6/predict_three_v9.py --model segformer --bundle . --rgb-dir INPUT_RGB --output predictions_segformer
```

DeepLab 先在 TF1.12 环境产生原始概率，再在当前 Python 环境应用冻结的后处理：

```bash
python source/paper_v6/predict_tf_v9.py --weights deeplab --vendor vendor --rgb-dir INPUT_RGB --output deeplab_probabilities
python source/paper_v6/predict_three_v9.py --model deeplab --bundle . --rgb-dir INPUT_RGB --probability-dir deeplab_probabilities --output predictions_deeplab
```

这两行使用两个不同 Python 环境。输入可为 PNG/JPEG/TIFF；文件名主体不能重复。任意图像独立推理，小于网络窗口的图像临时边缘填充，输出恢复原像素大小。输入的 RGB 不缩放、不改色，透明根图保留根像素原颜色。此轮仅根/背景分割；不会把 T3/T4 预测当作时序新生、持续、延伸或死亡真值。先只用于0353-2-B3，其他组预测需在当前验证之外另行检查。

## 重复本轮实际微调

训练从此前冻结网络开始，每个模型8轮、每轮32个原像素256图块，128次参数更新。`data/dataset/manifest.csv` 保存所有256次训练裁剪及增强参数（189个不同时期/坐标组合），并保存验证和C4的256/384两套原像素窗口。训练的256个抽样包含重复；这些数字不是独立原始图像数。三个模型复用相同 Python 随机采样序列；模型构造不改变该随机状态，原采样函数与导出函数逐数组比对256次通过。

起始权重不是本轮最终权重；复现微调须另取已发布不可变版本：

- U-Net：v6模型包的 `unet_pretrained/best.pt`，SHA256 `0d1def3aaa692864216988d8db4aa39bba5129dca1e69435a652a8140bed1f16`。
- SegFormer：v5模型包的 `segformer/best.pt`，SHA256 `9e819df7144626f9fc9c56a137fff678a12aebddf025aff939858886d9eecc7a`。
- DeepLab：v5模型包的 `deeplab/best_weights_fp16.npz` 与 `weight_names.json`，SHA256 `0e0b913e69366914392937fe0350e309c0f10e994218bf0a2bb0483829caa703`。

[v6起始权重包](https://raw.githubusercontent.com/Annalee811/LI/aa84f9a64f48d252b2e41e770915bfa64f440e55/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/paper-refinement-v6/0353-2-B3_refined_models_v6.zip)；[v5起始权重包](https://raw.githubusercontent.com/Annalee811/LI/e068b2d6d503135da6271a52303fca3826260887/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/root-background-stage-v5/0353-2-B3_root_background_v5_models.zip)。先校验上述单模型 SHA，再作为 `--initial`。DeepLab还必须保留同目录的变量名称表。

```bash
python source/paper_v6/refine_three_v9.py --data data --run NEW_RUN --model unet --initial INITIAL_UNET/best.pt
python source/paper_v6/refine_three_v9.py --data data --run NEW_RUN --model segformer --initial INITIAL_SEGFORMER/best.pt
# 以下在TF1.12环境运行
python source/paper_v6/refine_tf_v9.py --data data --run NEW_RUN --vendor vendor --initial INITIAL_DEEPLAB/best_weights_fp16.npz
# 回到Python3.12；三个网络的训练与推理结束后才运行
python source/paper_v6/evaluate_three_v9.py --data data --run NEW_RUN
```

网络轮次与阈值只根据 C3 校准 Dice 选择，epoch0 可胜出。最终连接器用 C0–C2 人工连通关系训练，连接/小噪点规则按相同 C3 网格选取；三模型胜者在读取 C4评价之前冻结。每模型都保留训练记录和全网格，包括无提升尝试。C4反复使用过，所以只称同位置评价，外部独立测试图像为0；不应用其分数继续选择本轮配置。

连接器仅补短距离、方向一致、有 RGB 模型概率支持的路径，并禁止接触第三个无关组件。不会对整张根掩膜膨胀或用人工真值补根。另外比较了0/0.025/0.05概率差的滞后恢复：只从高置信度根像素沿弱概率支持区域生长，验证结果决定是否启用。本轮三个模型最终均选0，即不启用该项。小噪点只考虑短小、不细长、概率不高的孤立分量；已连接的对象受保护。真值上的新增/剔除根及背景像素数同时报告。原始概率浮点数组、严格像素掩膜、所有新增路径 JSON 允许重算结果，24张正式评价图通过像素级复现检查。

## 评估与图片

Precision、Recall、Dice、IoU来自原尺寸整列融合预测，重叠小块只计一次。原始网络、微调选优网络、最终后处理分开报告。根长是人工/预测二值笔迹的骨架长度，单位像素，没有物理标尺或独立人工测长；三个模型分别提供R²一致性、RMSE、CCC和Bland–Altman。R²=1−SSE/SST，可为负值；OLS拟合线与一致性R²不同。8个空间块跨4期的32配对不是32个独立生物重复。

所有论文图保留600dpi PNG/TIFF和PDF，定量图另有SVG；图注随包保存。原始像素输出提供实际分辨率，600dpi排版不会增加光学清晰度。四期整图含训练区，不作为测试成绩。保存的死亡/褐变等人工属性不用于本轮二值训练，也不输出这些类别预测。

`*_centerline_display.svg` 是单独的平滑中心线示意，最大位移不超过0.5原像素；不是像素分割掩膜或真实根径图，不参与Dice或根长计算。`*_probability_uint16.png` 将0–1概率映射为0–65535供浏览，完整评估和重算使用float32 NPY。
