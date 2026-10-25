import hashlib,json,shutil,re
from pathlib import Path

ROOT=Path('/workspace/LI')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_editor_interaction_v53_20261025'
PREV=ROOT/'results/B3_root_structure_review_v52_20261025'

def main():
    html=(PREV/'annotate.html').read_text()
    old=(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_structure_v52/editor.js').read_text()
    assert html.count(old)==1
    html=html.replace(old,(HERE/'editor.js').read_text())
    draft=json.loads((OUT/'root_structure_draft.json').read_text())
    payload=json.dumps(dict(original=draft),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S)
    assert n==1
    style='''<style id="editingStyles53">
#editingToolbar53{flex:none;background:#edf5f1;border:1px solid #bad1c3;border-radius:6px;padding:6px;margin:5px 0;font-size:13px}
#editingToolbar53 button{font-size:13px;padding:5px 9px;margin:2px}#editingToolbar53 button.active{background:#285c42;color:white}
#editingToolbar53 .quickId53{display:flex;align-items:center;gap:8px;flex-wrap:wrap}#rootQuickId53{width:140px!important;padding:4px}
#editHelp53{font-weight:bold;margin:4px 0}#visibleStatus53{font-size:12px;color:#174d36;overflow-wrap:anywhere}
[hidden]{display:none!important}body.structure-pixel52 #structureControls52{max-height:24vh}
body.structure-pixel52 #b3QuickTools{flex:1;min-height:180px;overflow:auto}body.structure-pixel52 #b3OtherControls{flex:none;max-height:22vh}
</style>'''
    html+='\n'+style+'\n<script>\n'+(HERE/'interaction.js').read_text()+'\n</script>\n'
    (OUT/'annotate.html').write_text(html)
    test=html
    catalog=json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text())
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    for e in catalog:
        p=assets/e['name']
        if not p.exists():p.symlink_to(e['local_path'])
        test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
    (OUT/'annotate_test.html').write_text(test)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
    report={n:dict(bytes=(OUT/n).stat().st_size,sha256=hashlib.sha256((OUT/n).read_bytes()).hexdigest()) for n in ['annotate.html','root_structure_draft.json','source_progress_10.json']}
    assert draft['images']==json.loads((PREV/'root_structure_draft.json').read_text())['images']
    (OUT/'build_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
