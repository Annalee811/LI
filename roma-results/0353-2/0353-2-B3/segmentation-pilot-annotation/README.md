# 当前：四期完整裁剪拼接RGB分别标注

每期一张完整RGB：T1、T2、T3、T4，分别标注。先标整图，再裁小图并按空间分组划分数据集。图像保持2541×1867原始像素与颜色。

[先下载旧页标注备份助手](full-image-annotation-v1/0353-2-B3_preserve_existing_annotations.zip) · [下载四期完整图标注包](full-image-annotation-v1/0353-2-B3_four_full_RGB_annotation_v1.zip) · [放大标注工具栏预览](full-image-annotation-v1/toolbar_preview.png)

已有8张标注尚在用户本机旧页面内，云端未收到实际笔画。不要关闭、刷新旧页面或清理临时解压目录。先用备份助手在旧页保存全部JSON，再导入新版annotate.html。旧小图与四张完整RGB逐像素核验相同，按原坐标迁回；冲突保留当前值并报告，原备份全文随项目保留。

新版“放大标注”把画笔、擦除、撤销、粗细和时期切换固定在图旁边。Ctrl+Z撤销，Ctrl+S保存完整JSON；支持本机草稿恢复，移动文件或换浏览器仍须JSON。详见[使用说明](full-image-annotation-v1/README.md)。

105项浏览器检查使用另行构造的测试笔画；不代表收到或确认了用户实际8张标注，也不是根系识别准确率。后续分割需要实际完整标注与复核；没有模型训练或已取得的Dice等指标。白色分隔带不是图像数据；首次显露、遮挡、褐变不能单独证明新生或死亡。

## 历史结果

[two-window-review-v3 八张重标复核结果](two-window-review-v3/README.md)保留；更早baseline-review-v2与reviewable-annotations识别存在质量问题，不能直接当合格人工真值。旧分区清单用于坐标迁移，实际训练划分在整图标注完成后进行，同位置四期留在同一集合。仅0353-2-B3一个位置不能证明跨位置泛化。
