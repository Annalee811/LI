import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_root_id_edit_fix_v55_20261025';PREV=ROOT/'results/B3_root_family_review_v54_20261025'

def main():
    assert (OUT/'root_structure_draft.json').read_bytes()==(PREV/'root_structure_draft.json').read_bytes()
    html=(PREV/'annotate.html').read_text()
    html=html.replace("const types52={tip:","const types52={cut:'手动断开点',tip:").replace("const signs52={tip:","const signs52={cut:'|',tip:")
    css='<style id="numberEditStyles55">#numberEditNotice55{border-top:1px solid #bdcec2;padding-top:5px;margin-top:5px;font-size:12px}#draftStatus55{font-weight:bold;color:#184d34;margin:4px 0}#rootQuickId53,#rootId52{background:white!important;color:#111!important;pointer-events:auto!important;user-select:text!important}</style>'
    html+='\n'+css+'\n<script>\n'+(HERE/'id_edit.js').read_text()+'\n</script>\n<script>\n'+(HERE/'root_cut.js').read_text()+'\n</script>\n';(OUT/'annotate.html').write_text(html)
    test=html;catalog=json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text());(OUT/'assets').mkdir(exist_ok=True)
    for e in catalog:
        p=OUT/'assets'/e['name']
        if not p.exists():p.symlink_to(e['local_path'])
        test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
    (OUT/'annotate_test.html').write_text(test)
    for p in HERE.glob('*'):
        if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
    info={n:dict(bytes=(OUT/n).stat().st_size,sha256=hashlib.sha256((OUT/n).read_bytes()).hexdigest()) for n in ['annotate.html','root_structure_draft.json','source_progress_10.json']}
    (OUT/'build_verification.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
