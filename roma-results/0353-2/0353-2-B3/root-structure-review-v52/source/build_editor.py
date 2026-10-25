"""Extend the identity-safe editor; reuse immutable original RGB download assets."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path('/workspace/LI')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_root_structure_review_v52_20261025'


def main():
    html=(ROOT/'results/B3_annotation_restore_v50_20261025/annotate_restore_fixed_lite.html').read_text()
    draft=json.loads((OUT/'root_structure_draft.json').read_text())
    versions=dict(original=draft)
    text=json.dumps(versions,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+text+m[2],html,flags=re.S)
    assert n==1
    html=html.replace('<button data-reference-version="early">查看更早未完成稿（10-16）</button>','')
    html=html.replace('查看修订前原始人工稿（10-17）','重新载入本次结构初稿')
    html=re.sub(r'(<div class="tip" id="restoreNoticeV50">).*?(</div>)',r'\1默认载入本次 (10).json 及四期逐根结构初稿，原像素标注保持不变。结构均待人工复核。\2',html,flags=re.S)
    css='''<style id="structureStyles52">
body.b3-zoom60 main{grid-template-columns:365px minmax(0,1fr)}
#structureControls52{flex:1;min-height:0;overflow:auto;font-size:13px;padding:0 8px 12px 0}
#structureControls52 h2{font-size:17px;margin:4px 0}#structureControls52 label{margin:6px 0}
#structureControls52 button{font-size:12px;padding:5px 7px;margin:2px}
#structureControls52 select,#structureControls52 input[type=text]{max-width:100%;width:100%;font-size:13px}
#structureControls52 .inline52{display:inline-block;margin-right:10px}#structureControls52 details{border-top:1px solid #d5ded8;padding:7px 0}
#structureControls52 summary{cursor:pointer;font-weight:bold}#summary52,#rootInfo52,#nodeInfo52{font-size:12px;overflow-wrap:anywhere}
#segments52{height:110px}#comparison52{flex:none;font-size:12px;padding-top:4px}
#crops52{display:flex;gap:8px}#crops52>div{flex:1;max-width:165px}#crops52 b{display:block}
#crops52 canvas{width:100%;max-height:125px;object-fit:contain;background:#333}
body:not(.structure-pixel52) #b3QuickTools,body:not(.structure-pixel52) #b3OtherControls{display:none}
body.structure-pixel52 #structureControls52{flex:none;max-height:35vh}#zoom52{width:100%}
@media(max-width:850px){body.b3-zoom60 main{grid-template-columns:280px minmax(0,1fr)}#crops52{gap:3px}}
</style>'''
    html+='\n'+css+'\n<script>\n'+(HERE/'editor.js').read_text()+'\n</script>\n'
    (OUT/'annotate.html').write_text(html)
    catalog=json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text())
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    test=html
    for item in catalog:
        p=assets/item['name']
        if not p.exists():p.symlink_to(item['local_path'])
        test=test.replace(item['url'],'http://127.0.0.1:8894/assets/'+item['name'])
    (OUT/'annotate_test.html').write_text(test)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
    report={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [OUT/'annotate.html',OUT/'root_structure_draft.json',OUT/'source_progress_10.json']}
    (OUT/'build_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
