#!/usr/bin/env python3
"""Build a standalone, local-image annotation editor without dataset restrictions."""
from pathlib import Path
import argparse
import zipfile

HERE = Path(__file__).resolve().parent

def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    core = (HERE / 'annotate_custom_core.html').read_text()
    patch = (HERE / 'legacy_zoom60_patch.js').read_text()
    patch = patch.replace('0353-2-B3 人工根系标注 · 60倍操作版', '根系标注 · 自选图片 · 60倍')
    # Match the period in arbitrary filenames/extensions, e.g. T1_multiregion_RGB.png.
    patch = patch.replace("/_T[1-4](?=\\.png$)/i,'_T'+p", "/T[1-4](?!\\d)/i,'T'+p")
    patch = patch.replace("/_T[1-4](?=\\.png$)/i,'_T'+b.dataset.period", "/T[1-4](?!\\d)/i,'T'+b.dataset.period")
    begin = patch.index(' function updatePeriods(){')
    end = patch.index('\n sizeStage=', begin)
    patch = patch[:begin] + " function updatePeriods(){for(const b of periodButtons){const f=files[index],name=f?.name.replace(/T[1-4](?!\\d)/i,'T'+b.dataset.period);b.disabled=!f||!/T[1-4](?!\\d)/i.test(f.name)||!files.some(f=>f.name===name)}}" + patch[end:]
    page = core.replace('</script></html>', '</script><script>\n' + patch + '\n</script></html>')
    assert patch in page
    assert 'expectedImages' not in page and 'verified=' not in page
    (output / 'annotate.html').write_text(page)
    readme = '''根系标注工具：自选图片版

1. 将压缩包完整解压，在 Chrome 或 Edge 中打开 annotate.html。
2. 点“或选择一张／多张图片”，选择你自己的四期完整 RGB 或任意图片；也可选文件夹。
3. 不再内置或限定原来的60张小图，也不限定文件名与像素尺寸。
4. 选择分类颜色后沿实际根宽描画；Ctrl+Z 撤销。T1 均为已有基线根，后期判断需对照前期。
5. 最高60倍显示；切图保留倍率和原图像素位置。工具栏在图片旁。空格拖动或点“平移”。
6. 不需要画完即可点“保存全部标注进度 JSON”（或 Ctrl+S）。关闭前务必保存并保留原图。
7. 继续时重新选择相同原图，再选“恢复已保存的标注进度”中的 JSON。
8. 标注完成后填写标注者、勾选整图复核，可导出本张掩膜和属性 ZIP。掩膜与原图同尺寸。

原图不被修改或放大重采样；放大仅改变显示比例。进度 JSON 不包含 RGB，须保留原图。
同名但内容不同的图片各自保存，互不混用。跨期对比需要自行选择已经配准的 RGB；本工具不执行配准。
此前版本的进度格式不同，本版恢复功能接受本版导出的 JSON。
'''
    (output / '使用说明.txt').write_text(readme, encoding='utf-8-sig')
    archive = output / 'root_annotation_custom_images_v1.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['annotate.html', '使用说明.txt']:
            z.write(output / name, name)
    print(archive, archive.stat().st_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
