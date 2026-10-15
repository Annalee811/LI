# 当前：自选图片标注版

[下载离线标注工具 ZIP](custom-images-v1/root_annotation_custom_images_v1.zip)

解压后在 Chrome / Edge 打开 annotate.html，自行选择完整 RGB 或裁剪图即可标注。已取消原来60张图块的名称、哈希和448×448尺寸限制；没有预置图片。

保持原图尺寸的画笔与掩膜导出，最高60倍显示，工具栏固定在图旁，切图保留倍率与原图坐标。Ctrl+Z 撤销，Ctrl+S 下载全部标注进度 JSON，未完成也可保存。重新打开时选择相同原图，再导入本版 JSON 恢复。JSON 不含 RGB，须保留原图。

浏览器验证包含大尺寸和非正方形图片的实际指针绘制、撤销、跨期切换、同名不同内容隔离、进度恢复及原尺寸导出。验证仅涉及工具功能和数据完整性。

旧工具与旧图包保留在 legacy-zoom60-v1/、full-image-annotation-v1/ 和 two-window-review-v3/。
