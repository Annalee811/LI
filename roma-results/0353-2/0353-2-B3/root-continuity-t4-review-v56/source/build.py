"""Build editor on existing V55 controls, with native N- root-label support."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_root_continuity_review_v56_20261026'
def main():
 html=(ROOT/'results/B3_root_id_edit_fix_v55_20261025/annotate.html').read_text()
 hierarchy=(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_family_v54/hierarchy.js').read_text();new=hierarchy
 start=new.index('function parseLabel54(');end=new.index('function rawHierarchy54()',start)
 new=new[:start]+'''function parentLabel56(label){const base=label.match(/^(?:N-)?\\d{4,6}/)?.[0];if(!base||base.length===label.length)return null;return label.slice(0,label.lastIndexOf('-'));}
function parseLabel54(value){let s=String(value).trim().toUpperCase().replace(/^R/,'');if(/^\\d+$/.test(s))s=s.padStart(4,'0');if(!/^(?:N-)?\\d{4,6}(?:-N?[1-9]\\d*)*$/.test(s))throw Error('编号格式为 0001、0001-1、0001-N1 或独立新根 N-0001；N-0001-1 为其侧根。');return s;}
'''+new[end:]
 new=new.replace("h.label.includes('-')","parentLabel56(h.label)!==null")
 new=new.replace("const at=label.lastIndexOf('-');let parent=null,kind='independent';if(at>=0){parent=resolve54(label.slice(0,at));kind=label.slice(at+1).startsWith('N')?'new':'existing';", "const parentName=parentLabel56(label);let parent=null,kind='independent';if(parentName){parent=resolve54(parentName);kind=label.slice(parentName.length+1).startsWith('N')?'new':'existing';")
 new=new.replace("if(label.includes('-'))resolve54(label.slice(0,label.lastIndexOf('-')));","if(parentLabel56(label))resolve54(parentLabel56(label));")
 assert html.count(hierarchy)==1;html=html.replace(hierarchy,new)
 idjs=(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_id_edit_v55/id_edit.js').read_text();updated=idjs.replace("const parts=text.split('-');if(/^\\d+$/.test(parts[0]))parts[0]=parts[0].padStart(4,'0');\n for(let i=1;i<parts.length;i++)", "const parts=text.split('-'),base=parts[0]==='N'?1:0;if(/^\\d+$/.test(parts[base]||''))parts[base]=parts[base].padStart(4,'0');\n for(let i=base+1;i<parts.length;i++)")
 assert updated!=idjs and html.count(idjs)==1;html=html.replace(idjs,updated)
 cut=(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_id_edit_v55/root_cut.js').read_text();updated=cut.replace("else if(!target&&desired.includes('-'))resolve54(desired.slice(0,desired.lastIndexOf('-')));","else if(!target&&parentLabel56(desired))resolve54(parentLabel56(desired));")
 assert updated!=cut and html.count(cut)==1;html=html.replace(cut,updated)
 draft=json.loads((OUT/'root_structure_draft.json').read_text());payload=json.dumps(dict(original=draft),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="reference-original-versions-v50" type="application/json">).*?(</script>)',lambda m:m[1]+payload+m[2],html,flags=re.S);assert n==1
 html+='\n<script>\n'+(HERE/'continuity.js').read_text()+'\n</script>\n';(OUT/'annotate.html').write_text(html)
 test=html;(OUT/'assets').mkdir(exist_ok=True)
 for e in json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text()):
  p=OUT/'assets'/e['name']
  if not p.exists():p.symlink_to(e['local_path'])
  test=test.replace(e['url'],'http://127.0.0.1:8894/assets/'+e['name'])
 (OUT/'annotate_test.html').write_text(test)
 for p in HERE.iterdir():
  if p.is_file():shutil.copy2(p,OUT/'source'/p.name)
 info={n:dict(bytes=(OUT/n).stat().st_size,sha256=hashlib.sha256((OUT/n).read_bytes()).hexdigest()) for n in ['annotate.html','root_structure_draft.json','source_progress_12.json']}
 (OUT/'build_verification.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
