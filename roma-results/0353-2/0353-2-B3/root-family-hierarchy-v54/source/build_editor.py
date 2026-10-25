import hashlib,json,re,shutil
from pathlib import Path

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_root_family_review_v54_20261025'
PREV=ROOT/'results/B3_editor_interaction_v53_20261025'

def main():
    html=(PREV/'annotate.html').read_text()
    for name in ['editor.js','interaction.js']:
        old=(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/editor_interaction_v53'/name).read_text()
        assert html.count(old)==1;html=html.replace(old,(HERE/name).read_text())
    draft=json.loads((OUT/'root_structure_draft.json').read_text());payload=json.dumps(dict(original=draft),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S);assert n==1
    style='''<style id="familyStyles54">#familyPanel54{border:1px solid #c2d4ca;background:#f2f7f4;padding:7px;margin:7px 0}#familyPanel54 button{font-size:12px}#familyMembers54 button{max-width:100%;overflow-wrap:anywhere}#familyInfo54{font-size:12px;overflow-wrap:anywhere}#familyLegend54{font-size:12px;font-weight:bold;margin-top:3px}#rootQuickId53{width:190px!important}.inline53{font-size:12px}</style>'''
    html+='\n'+style+'\n<script>\n'+(HERE/'hierarchy.js').read_text()+'\n</script>\n'
    (OUT/'annotate.html').write_text(html)
    catalog=json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text());test=html;(OUT/'assets').mkdir(exist_ok=True)
    for e in catalog:
        p=OUT/'assets'/e['name']
        if not p.exists():p.symlink_to(e['local_path'])
        test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
    (OUT/'annotate_test.html').write_text(test)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
    assert draft['images']==json.loads((OUT/'source_progress_10.json').read_text())['images']
    info={n:dict(bytes=(OUT/n).stat().st_size,sha256=hashlib.sha256((OUT/n).read_bytes()).hexdigest()) for n in ['annotate.html','root_structure_draft.json','source_progress_10.json']}
    (OUT/'build_verification.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))

if __name__=='__main__':main()
