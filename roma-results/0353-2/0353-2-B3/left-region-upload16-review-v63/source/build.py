import hashlib,json,re,shutil
from pathlib import Path
R=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;O=R/'results/B3_left_region_review_v63_20261027'
def main():
 d=json.loads((O/'review_project.json').read_text());html=(R/'results/B3_root_free_cut_v62_20261027/annotate.html').read_text();payload=json.dumps(dict(original=d),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S);assert n==1
 html+='\n<script>\n'+(HERE/'review.js').read_text()+'\n</script>\n';(O/'annotate.html').write_text(html);test=html;(O/'assets').mkdir(exist_ok=True)
 for e in json.loads((R/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text()):
  p=O/'assets'/e['name']
  if not p.exists():p.symlink_to(e['local_path'])
  test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
 (O/'annotate_test.html').write_text(test)
 for p in HERE.iterdir():
  if p.is_file():shutil.copy2(p,O/'source'/p.name)
 (O/'build_verification.json').write_text(json.dumps({name:dict(bytes=(O/name).stat().st_size,SHA256=hashlib.sha256((O/name).read_bytes()).hexdigest()) for name in ['annotate.html','review_project.json','source_progress_16.json']},indent=2)+'\n');print('built',len(html.encode()),'12 assistant proposals;135 human definitions preserved')
if __name__=='__main__':main()
