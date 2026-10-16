# 0353-2-B3 上部 T1/T2 分割试训

结果说明请先看 `结果说明.txt`。所有图像切块和标签均来自用户提供的标注，未重新绘制、膨胀或生成伪标注。下方区域不参与。

复现已有数据的训练（在解压目录运行）：

```bash
python -m pip install -r requirements.txt
python source/train_uploaded_upper.py --existing-dataset --output .
```

默认自动检测 CUDA，CPU 使用4个线程。固定随机种子；不同设备或PyTorch实现仍可能产生数值差异。复训会替换当前目录的训练、预测和评估结果，请先复制完整目录。

校验：`python source/verify_uploaded_upper.py --run .`

本模型是从随机初始化开始的小型 U-Net，仅做根／背景分割。T2时序属性与其他属性均原样保留于数据中；本轮不训练这些分类。训练图块60张、验证20张、空间留出测试20张来自同一位置的两次原始采集，不能当作100次独立采集。

测试列不参与拟合、早停、选模型或选阈值。重叠图块融合后仅统计一次源像素。64像素空间隔离减少直接像素泄漏，但同一根系和同一背景仍有相关性，不构成独立地点验证。

未标记的细根在输入掩膜中为背景，模型与该掩膜的一致度不等于解剖学意义上的完整根系识别率。DeepLabV3+结果见model_comparison.csv及deeplab/；RootPainter尚未训练；无独立人工根长测量，不报告其R²或RMSE。

上传模型的权重单独保存在DeepLabV3plus_TF112_model.zip，数据与结果包包含其指标及预测；两份ZIP可分别解压。
