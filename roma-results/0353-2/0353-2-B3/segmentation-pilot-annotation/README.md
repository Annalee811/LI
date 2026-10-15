# 最新：两组四期，共8张标注复核版3

[下载可编辑标注包](two-window-review-v3/0353-2-B3_two_groups_8_images_v3.zip)

[第一组：B3_A0_C0 四期原图与标注](two-window-review-v3/review_panels/B3_A0_C0_original_and_roots.png) · [第二组：B3_A0_C1 四期原图与标注](two-window-review-v3/review_panels/B3_A0_C1_original_and_roots.png)

解压后打开 annotate.html，8张图像与标注已内置，可修改、Ctrl+Z撤销、按编号查看四期和保存全部项目JSON。T1全部已标根均为已有基线根；来源不明与跨期对应不确定的段保留未定。原始RGB不变。

这是首轮复核结果，边界、细根完整性、粗细及对应关系仍需检查，尚未形成独立人工真值。软件检查不表示根系识别准确率。详见[本版说明](two-window-review-v3/README.md)。

## 历史版本

baseline-review-v2/ 为此前四张示范重描与其他56张清理结果；reviewable-annotations/ 为更早60张预填版本。其识别质量已被指出存在问题，保留用于追溯，不能直接当作合格人工精标。当前入口 annotate.html 仅包含本次8张。

## 后续分割预实验

原始60张图块来自0353-2-B3同一位置，共15个空间窗口四期；固定计划为训练36、验证12、空间留出测试12张。同窗四期不跨集合；这不代表其他位置的独立测试。当前没有完成合格人工真值、模型训练或真实分割指标。不能把示例Dice 0.956当作已取得结果。

RootPainter、U-Net和DeepLabV3+比较需先完成标注复核。颜色、来源批次、当前时序状态和存活属性分别保存；消失、遮挡或褐变不能单独证明死亡。根长物理单位还需要标尺，独立一致性评价需要另行人工测量。
