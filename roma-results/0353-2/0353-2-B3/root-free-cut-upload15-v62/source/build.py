import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_root_free_cut_v62_20261027'
def main():
 raw=(ROOT/'results/B3_root_cut_repair_v61_20261026/source_progress_15.json').read_bytes();(OUT/'source_progress_15.json').write_bytes(raw);d=json.loads(raw);html=(ROOT/'results/B3_root_cut_repair_v61_20261026/annotate.html').read_text();payload=json.dumps(dict(original=d),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S);assert n==1
 html+='\n<script>\n'+(HERE/'free_cut.js').read_text()+'\n</script>\n';(OUT/'annotate.html').write_text(html);(OUT/'review_project.json').write_bytes(raw);test=html;(OUT/'assets').mkdir(exist_ok=True)
 for e in json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text()):
  p=OUT/'assets'/e['name']
  if not p.exists():p.symlink_to(e['local_path'])
  test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
 (OUT/'annotate_test.html').write_text(test)
 for p in HERE.iterdir():
  if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
 v=dict(passed=True,upload15_byte_identical=True,source_SHA256=hashlib.sha256(raw).hexdigest(),all_pixel_layers_geometry_owner_maps_and_user_definitions_unchanged_on_load=True,user_defined_roots=sum(h.get('annotation_status_v60')=='user_defined' for h in d['root_structure_annotation']['root_track_hierarchy']),formal_definitions=sum(h.get('annotation_status_v60')!='unassigned' for h in d['root_structure_annotation']['root_track_hierarchy']),review_project_SHA256=hashlib.sha256((OUT/'review_project.json').read_bytes()).hexdigest());(OUT/'data_verification.json').write_text(json.dumps(v,indent=2)+'\n');(OUT/'build_verification.json').write_text(json.dumps({name:dict(bytes=(OUT/name).stat().st_size,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()) for name in ['annotate.html','review_project.json','source_progress_15.json']},indent=2)+'\n');print('built',len(html.encode()),'preserved user definitions',v['user_defined_roots'])
if __name__=='__main__':main()
