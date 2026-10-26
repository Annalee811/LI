import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_root_temporal_review_v59_20261026'
def main():
 html=(ROOT/'results/B3_root_parentage_v58_20261026/annotate.html').read_text();draft=json.loads((OUT/'root_structure_draft.json').read_text());payload=json.dumps(dict(original=draft),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S);assert n==1
 # Older saved V58 HTML kept a runtime-built parentage panel. Remove it on
 # replay before creating fresh bound controls, just like the other panels.
 html=html.replace("const parentPanel58=document.createElement('div');","document.getElementById('parentagePanel58')?.remove();const parentPanel58=document.createElement('div');")
 html+='\n<script>\n'+(HERE/'temporal.js').read_text()+'\n</script>\n';(OUT/'annotate.html').write_text(html);test=html;(OUT/'assets').mkdir(exist_ok=True)
 for e in json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text()):
  p=OUT/'assets'/e['name']
  if not p.exists():p.symlink_to(e['local_path'])
  test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
 (OUT/'annotate_test.html').write_text(test)
 for p in HERE.iterdir():
  if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
 (OUT/'build_verification.json').write_text(json.dumps({n:dict(bytes=(OUT/n).stat().st_size,sha256=hashlib.sha256((OUT/n).read_bytes()).hexdigest()) for n in ['annotate.html','root_structure_draft.json','source_progress_14.json']},indent=2)+'\n');print('built', (OUT/'annotate.html').stat().st_size)
if __name__=='__main__':main()
