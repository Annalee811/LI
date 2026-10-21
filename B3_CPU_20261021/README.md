# B3 三模型 CPU 训练结果（本地待确认上传）

**本轮结果尚不能可靠识别新出现根和延伸段，不建议直接迁移到其他微根管。**

验证集选择 ChangeFormer：测试宏平均Dice=0.2412，持续段=0.6599，新出现根=0.0637，延伸段=0。

仅为单个B3点位的空间测试；实例身份、交叉归属和侧根父子关系未经人工实例真值验证。

## 文件索引

| 文件 | MiB |
|---|---:|
| [four_period_composite.tif](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/four_period_composite.tif) | 52.2 |
| [four_period_layers.tif](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/four_period_layers.tif) | 25.8 |
| [paper_figures.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/paper_figures.zip) | 32.1 |
| [all_results_weights_scripts_data.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/all_results_weights_scripts_data.zip) | 1173.2 |
| [training_data.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/training_data.zip) | 43.1 |
| [scripts_and_configs.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/scripts_and_configs.zip) | 11.6 |
| [weights_siamese_unet.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/weights_siamese_unet.zip) | 27.3 |
| [weights_stanet.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/weights_stanet.zip) | 299.8 |
| [weights_changeformer.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/weights_changeformer.zip) | 527.0 |
| [prediction_results.zip](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/prediction_results.zip) | 264.4 |
| [training_data_download_index.json](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/training_data_download_index.json) | 0.1 |
| [model_comparison.csv](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/model_comparison.csv) | 0.0 |
| [README_中文.md](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/README_中文.md) | 0.0 |
| [verification.json](https://github.com/Annalee811/LI/raw/refs/heads/b3-cpu-artifacts-20261021-v1/B3_CPU_20261021/verification.json) | 0.0 |

上传状态以远端分支是否存在为准；本地文件清单不表示已上传。

完整总包含三个模型权重、恢复检查点、完整脚本、数据、标注、切片及训练/预测/交叉检查结果。源项目路径和既有实验不在本提交中修改。
