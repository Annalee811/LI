# 人工标注完成后的运行

使用已配置的Python环境；本包的 `requirements-pilot.txt` 是额外依赖，不替换RoMa依赖或原项目锁文件。

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python /workspace/LI/.venv/bin/python -r requirements-pilot.txt
```

解压图包后以其根目录为工作目录，把人工标注ZIP解压到 `labels`。首先检查人工标注，未完成时该命令非零退出；不会使用自动候选或空白掩膜训练。

```bash
/workspace/LI/.venv/bin/python pilot.py check-labels --kit . --split train
/workspace/LI/.venv/bin/python pilot.py check-labels --kit . --split val
```

两个架构共享编码器及数据设置。以下从头训练示例只应在人工真值齐全后运行；若采用ImageNet预训练，为两个模型同时把 `--encoder-weights none` 改成 `imagenet`。记录策略，勿混合不同策略后归因于架构。

```bash
/workspace/LI/.venv/bin/python pilot.py train --kit . --model unet --encoder-weights none --output runs/unet
/workspace/LI/.venv/bin/python pilot.py train --kit . --model deeplabv3plus --encoder-weights none --output runs/deeplabv3plus
```

使用验证集选择停止轮数，固定测试策略后才预测留出测试。真实测试标签需要独立人工复核。

```bash
/workspace/LI/.venv/bin/python pilot.py predict --kit . --checkpoint runs/unet/best.pt --split val --output predictions/unet/val
/workspace/LI/.venv/bin/python pilot.py evaluate --kit . --predictions predictions/unet/val --split val --model U-Net --output evaluations/unet/val
/workspace/LI/.venv/bin/python pilot.py predict --kit . --checkpoint runs/unet/best.pt --split test --output predictions/unet/test
/workspace/LI/.venv/bin/python pilot.py evaluate --kit . --predictions predictions/unet/test --split test --model U-Net --output evaluations/unet/test
```

DeepLabV3+同样运行。缺少独立人工根长时分割指标仍可计算，根长一致性为空，不会用人工掩膜派生长度冒充人工测量。填入经过复核的 `manual_lengths.csv` 后重算评估，输出真实R²、RMSE、CCC等及根长图。

RootPainter官方训练导入：

```bash
/workspace/LI/.venv/bin/python pilot.py export-rootpainter --kit . --output rootpainter_project_input
```

用官方RootPainter软件完成实际训练，导出与小图同名的灰度二值PNG（背景0、根255），或导出0—1概率NPY；保持原始图块坐标和尺寸。预测不能由本包的未训练U-Net生成后改名。

```bash
/workspace/LI/.venv/bin/python pilot.py evaluate --kit . --predictions predictions/rootpainter/test --split test --model RootPainter --output evaluations/rootpainter/test
/workspace/LI/.venv/bin/python pilot.py compare --reports evaluations/unet/test/metrics.json evaluations/deeplabv3plus/test/metrics.json evaluations/rootpainter/test/metrics.json --output model_comparison_test.csv
```

比较脚本要求各方法使用完全相同的人工标签哈希和分组。验证/测试分开报告，图块数与物理观察窗数分别保留。真实实验未运行；0.956仅是用户给出的预期值，不能写入空缺结果。
