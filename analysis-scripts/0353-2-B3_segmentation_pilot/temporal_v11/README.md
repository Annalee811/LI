# B3 三模型时序分类：连接点与类别边界优化 v11

基于v10三套时序头继续训练，根背景主干、概率、阈值及根掩膜保持冻结。原始人工JSON和根/时序标签不修改，仅使用2541×1108上半部。T1基础根；T2相对T1、T3相对T2、T4相对T3。持续1紫、新生2绿、根尖新增的生长段3金黄、T1基础4蓝；背景0、未指定时序属性255排除评分。死亡不纳入。原有段不会因为根尖延伸而全部改成生长段。

优化包括：
- 在原11个输入通道后增加14个预测几何通道：局部重叠、邻域位移容许、正确骨架端点/分叉、方向一致性及方向匹配分数。25通道首层保留旧11通道权重，新增权重零初始化。
- 采用排除对角冗余边的骨架图度数，避免把普通弯折错当分叉。方向只作局部对应的学习线索，不作为根身份或出生的硬规则。
- 每轮48张原像素256配对裁剪，约25%以各模型预测连接区为中心、25%以训练人工类别交界为中心，其余为类别抽样和均匀采样。连接区坐标可因模型不同而不同，采样政策一致；全部真实抽样参数保留。
- 连接区交叉熵权重1+2×连接邻近度；增加人工类别交界的辅助预测损失，以及仅在相同人工类别相邻根像素上的轻量一致性损失。不同类别的交界不被强制平滑。输入RGB背景置零，连接不自动意味着同一类。

评估沿用根背景阶段的分组，避免冻结根模型参与过训练的区域被冒称整条流水线的新验证折：C0–C2拟合，C3选择，C4为已反复查看的同位置基准。四期对应空间始终同组，64px分隔。原始图4张、相邻转换3组；网格216对训练、72对验证、72对基准是相关切块，不是独立个体。独立跨管标注测试图0。

评估头只从v10 evaluation_head.pt开始，不能从全图训练过的final_head.pt开始。选择分数为0.6×全区三类宏Dice+0.4×连接区三类宏Dice，同时全区宏Dice和每一类Dice不能比初始模型下降超过0.005。每5轮检查，最多60轮、至少40轮，6次无改善停止；允许保留epoch0。训练期间原v10类别概率乘数冻结。完成训练后，针对用户强调的类别互误，再在C3对63组额外概率乘数进行校准，并允许选择原epoch0。硬约束为：连接区和前期根尖区的持续↔新生、持续↔生长、新生↔生长六项互误均不得高于v10；整体及每类Dice下降不超过0.005，连接区宏Dice下降不超过0.005。在合格方案中先最小化六项互误总数，再最大化全区宏Dice。C4不参与该选择。全部校准网格、校准前权重和指标均保存；这属于验证集开发，不是未见精度证明。另用v10 final_head.pt在全部上半部继续训练所选额外轮数，包含原分隔区；这是部署拟合头，不用于未见精度报告。若选中epoch0，则最终头保留旧模型的等价扩展，而非声称提升。

连接区定义为同一冻结v9预测根掩膜的正确骨架分叉12px邻域；根尖区是前期预测端点20px邻域。旧新版始终使用同一ROI。另列新生↔持续、持续↔生长、新生↔生长误分，及每类Precision/Recall/Dice/IoU。C3参与模型选择，C4反复查看，两者都不等于跨根管泛化验证。无真值和预测的类别指标为空，出现误报则Dice/IoU为0；不以背景占比很高的准确率替换根类指标。

完整四期图是全标注拟合展示；留出评估图另存。图的600dpi格式不表示分类已达到论文结论所需精度。根长为人工/预测掩膜骨架的原像素长度，无独立人工尺量、无毫米标尺。R²=1−SSE/SST，保留负值；相关块跨期重复，不是独立生物学重复。新近可见并不能排除遮挡解除，交叉也不能证明物理连接。

复现（自动检测CUDA；当前实验CPU BF16计算，参数与损失FP32）：

```bash
python source/temporal_v11/run_all.py --data RUN/data --roots ROOT_V9 --previous PREVIOUS_V10 --output RUN --epochs 60 --samples 48
python source/temporal_v11/calibrate.py --run RUN --roots ROOT_V9 --previous PREVIOUS_V10
python source/temporal_v11/deliver.py --run RUN --roots ROOT_V9 --products
python source/temporal_v11/verify.py --run RUN --roots ROOT_V9
python source/temporal_v11/compare.py --run RUN --previous PREVIOUS_V10
python source/temporal_v11/export_focus.py --run RUN
MPLCONFIGDIR=/tmp/B3_v11_mpl XDG_CACHE_HOME=/tmp/B3_v11_cache python source/temporal_v11/figures.py --run RUN
MPLCONFIGDIR=/tmp/B3_v11_mpl XDG_CACHE_HOME=/tmp/B3_v11_cache python source/temporal_v11/figures_compare.py --run RUN --previous PREVIOUS_V10
```

新数据包提供未改动原标注、四期上半部、对应掩膜、三模型连接点训练实际前96对窗口及剔背景根RGB，并保存增强参数。训练读取上半部和真实抽样，不依赖人工重新标注。冻结第一阶段概率和DeepLab根权重复用v10下载包；v10全部新旧时序起始权重包含在v11权重脚本包的previous_v10中。源first-stage权重和代码也保留。不要混用不同模型的根掩膜或前期概率。

对其他同尺寸、已配准位置，可先执行对应根背景模型，再使用predict.py传入该模型final_head.pt、T1_RGB.png…T4_RGB.png和Tn_probability.npy/Tn_root_mask.png。时序输入仅使用预测根，不读取人工前期掩膜；输出保持原尺寸与原色根RGB，分类色图另存。类别置信度不是未标注位置的真实准确率。
