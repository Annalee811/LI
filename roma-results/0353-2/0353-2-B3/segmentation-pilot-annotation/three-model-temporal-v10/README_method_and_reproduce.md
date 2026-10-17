本实验在已有三套根系/背景分割结果上训练三个独立时序分类头。

模型分别为：U-Net（ResNet18）、用户上传的原始TensorFlow DeepLabV3+（ResNet-v2-50）、SegFormer-B0。第一阶段v9的权重、阈值与后处理全部冻结；第二阶段采用相同结构的小型U形时序CNN，分别接受对应第一阶段的结果。它不是三套主干的时序端到端微调。背景在进入时序头前已置零，只有保留的根RGB、预测根概率及前期预测根的端点/分支/距离线索进入时序头。

标签：T1全部为基础根（输出4）；T2相对T1、T3相对T2、T4相对T3。持续根输出1（紫）、前期未观察到的新生根输出2（绿）、原根尖新增的生长段输出3（黄）；背景0。原始人工时序值3/1/2分别映射为输出1/2/3。死亡不纳入；未指定时序属性的根像素为255，排除时序损失和评分。新近可见并不证明其生物学诞生时间；遮挡和配准误差也可能影响时序类别。

原始人工JSON及所有上半部标签保持不变。只取2541×1108上半部，不使用下半部。训练样本为原像素256×256相邻期对；同位置四期始终同组，重叠切块和增强不增加独立个体数。保留原空间分组，便于与冻结的第一阶段比较：训练C0–C2，验证C3，C4为已多次查看的同位置基准。每个时期区域之间有64像素间隔。前三个列区训练216对，C3验证72对，C4评估72对；原始图始终只有四张、三个相邻转换，外部独立标注测试图为0。

模型选择只使用C3端到端时序宏Dice。C4不参与本次拟合或选择，但它不是新独立测试集。给出端到端指标与“已正确检出且有人工时序标签的根像素上的分类”两种评分，后者不代表完整分割准确性。各类别支持量、逐期指标及原始混淆计数均保留。无人工支持时Recall为空；若该类仍有误报，则Dice/IoU为0；真值与预测都没有该类时Dice/IoU为空，不记为满分。所有主指标严格按原像素评估，没有扩大真值或改变人工值。

本次没有把单独轮换时序头称为五折无泄漏验证：冻结的根背景网络已使用C0–C2标注训练，轮换时这些区域不能成为整条流水线的未见验证区。因此继续沿用原分组作可比预实验，最终再使用全部上半部标注拟合部署头。重新对整条流水线执行空间交叉验证，需要各折重训根背景网络；这不作为本次追加时序头的前提，也不要求用户标注其他位置。

评估头训练后，锁定训练轮数，另行从相同初始化训练使用全部B3上半部的最终时序头，包含原留出区和间隔区。最终版本供后续预测使用；四期完整展示图是参与训练位置上的拟合展示，不能作为独立精度证据。第一阶段根背景网络仍是冻结v9版本。头不会恢复第一阶段漏掉的根；自动前期输入来自该模型自己的根分割，预测时不读取人工前期根掩膜。分类是像素级时序分类，不等于已验证的根对象身份追踪。

根长单位为原像素，由人工掩膜和预测掩膜各自骨架计算。没有独立人工尺量和物理标尺。R²为1−SSE/SST，以一比一对应评估一致性，保留负数；根长一致不代表空间分割一致。每区8个不重叠统计块跨期重复，属于相关描述性点，不能当作独立生物学重复。

运行（在仓库根目录，使用现有Python环境；自动检测CUDA；CPU支持时采用BF16计算、FP32参数与损失）：

```bash
python source/temporal_v10/prepare.py --source B3_root_background_v5 --output RUN/data
python source/temporal_v10/prepare_roots.py --data RUN/data --roots ROOT_V9
python source/temporal_v10/run_all.py --data RUN/data --roots ROOT_V9 --output RUN --epochs 100 --samples 48
python source/temporal_v10/calibrate.py --run RUN --roots ROOT_V9
python source/temporal_v10/deliver.py --run RUN --roots ROOT_V9 --products
python source/temporal_v10/verify.py --run RUN --roots ROOT_V9
MPLCONFIGDIR=/tmp/B3_temporal_mpl XDG_CACHE_HOME=/tmp/B3_temporal_cache python source/temporal_v10/figures.py --run RUN
```

数据包已经包含prepare.py的结果，可直接从prepare_roots或训练步骤开始。第一阶段概率包按原目录结构解压合并为ROOT_V9，训练将读取模型的postprocessing_selection.json及概率。若要从原RGB重建第一阶段结果，使用包内source/paper_v6/predict_three_v9.py；原TF模型先用predict_tf_v9.py在其TF1.12运行环境产生概率。现有第一阶段完整训练与运行说明见v9的README_reproduce.md。

其他位置预测：先对已配准、同尺寸的相邻RGB执行对应模型根/背景分割，得到各期Tn_probability.npy与Tn_root_mask.png，再调用：

```bash
python source/temporal_v10/predict.py --head MODEL/final_head.pt --rgb-dir RGB --root-dir ROOTS --output PREDICTION
```

RGB文件名为T1_RGB.png至T4_RGB.png。缺中间期时不能自动称为相邻转换；脚本会阻止未声明的跨期推断。输出保留原尺寸、原始根RGB颜色，另存分类图、透明根图、叠加图和条件类别置信度。置信度不是未经标注位置上的真实准确率。

类别校准：在原始验证最佳轮数锁定后，三个模型使用完全相同的30组有限网格，仅在C3选择新生/生长相对于持续根的概率乘数。校准前后指标和全部网格均保存，C4不参与选择。校准参数嵌入权重，同样用于最终头及单独预测。为使缓存与实时推断一致，融合后的条件概率先执行FP16往返转换，再校准；权重仍以FP32保存。
