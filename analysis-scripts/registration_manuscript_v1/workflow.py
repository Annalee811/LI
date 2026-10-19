"""Reproduce the published registration geometry, RGB, QC and four one-page figures.

Geometry is a frozen, semi-automatic manuscript selection. Reusing it is not a
new pose estimate. Historical estimation source is supplied for audit/rechecks.
"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
SUPPORT=HERE.parent/'all_series_localization_v2'
sys.path[:0]=[str(HERE.parent),str(SUPPORT)]
from dataset_identity import require_active_position,canonical_position,tube_id

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def registry():
 r=json.loads((HERE/'geometry_registry_current.json').read_text());assert len(r)==18
 for key,v in r.items():assert require_active_position(key)==key and v['position']==key
 return r
def expected():
 with (PROJECT/'dataset/current/active_images.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def call(script,*args):
 env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(SUPPORT/'vendor'),str(SUPPORT),str(HERE.parent),env.get('PYTHONPATH','')]);subprocess.run([sys.executable,str(script),*map(str,args)],env=env,check=True)
def link_or_copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(src)==sha(dst),('Existing file differs',dst);return
 try:dst.symlink_to(src.resolve())
 except OSError:shutil.copy2(src,dst)
def validate_raw(w):
 records=[]
 for row in expected():
  files=list((w/'inputs/raw'/row['position_id']).glob(row['period']+'.*'));assert len(files)==1,(row['position_id'],row['period'],'Missing or ambiguous input')
  actual=sha(files[0]);assert actual==row['original_SHA256'],('Original checksum mismatch',files[0]);records.append(dict(position=row['position_id'],period=row['period'],SHA256=actual))
 assert len(records)==66;return records

def prepare(w,raw_root):
 if raw_root is None:
  # Source releases retain historical IDs; only the active view below is used.
  call(SUPPORT/'prepare_inputs.py','--work',w/'source_releases');raw_root=w/'source_releases/inputs/raw'
 raw_root=raw_root.resolve();r=registry()
 for pos,entry in r.items():
  candidates=[raw_root/pos]
  if entry['legacy_source_id']!=pos:candidates.append(raw_root/entry['legacy_source_id'])
  src=next((p for p in candidates if p.is_dir()),None);assert src is not None,('Position not found',pos,raw_root)
  for t in entry['periods']:
   files=list(src.glob(f'T{t}.*'));assert len(files)==1,(pos,t)
   assert sha(files[0])==entry['input_sha256'][f'T{t}'];link_or_copy(files[0],w/'inputs/raw'/pos/files[0].name)
 records=validate_raw(w);assert len(list((w/'inputs/raw').iterdir()))==18
 js(w/'source_verification.json',dict(original_files=66,positions=18,canonical_alias_applied=True,excluded_position_absent=True,files=records));print('PREPARE_PASS: canonical 18 positions, 66 checksum-verified original files',flush=True)

def render(w):
 validate_raw(w)
 from render_results import render_position
 for e in registry().values():render_position(w,e,force=True)
 rows=expected();catalog=[]
 for pos,e in registry().items():
  folder=w/'deliverable'/pos;m=json.loads((folder/'geometry_metadata.json').read_text());rgb={};originals={}
  for t in e['periods']:
   key=f'T{t}';p=folder/f'RGB/{key}_independent_stitched_RGB.png';want=next(r['registered_RGB_SHA256'] for r in rows if r['position_id']==pos and r['period']==key)
   assert sha(p)==want,('Rendered RGB differs from published pixels',pos,key);rgb[key]=str(p.resolve());originals[key]=str(next((folder/'originals').glob(key+'*')).resolve())
  catalog.append(dict(position_id=pos,tube_id=tube_id(pos),legacy_source_id=e['legacy_source_id'],periods=e['periods'],RGB=rgb,originals=originals,folder=str(folder.resolve()),review_required=any(r['review_required'] for r in e['regions']),metadata=m))
 catalog.sort(key=lambda e:e['position_id']);js(w/'catalog.json',catalog);js(w/'RGB_reproduction_verification.json',dict(all_66_registered_RGB_SHA256_match_published=True,geometry_mode='Frozen published semi-automatic similarities; not new localization'))
 print('RENDER_PASS: all 66 registered RGB images match published pixel/file hashes',flush=True)

def figures(w):
 assert (w/'catalog.json').exists(),'Run render first'
 call(HERE/'figure_series.py','--catalog',w/'catalog.json','--output',w/'figures')
 call(HERE/'figure_B3.py','--catalog',w/'catalog.json','--raw-root',w/'inputs/raw','--output',w/'figures')
 from PIL import Image
 proofs=[]
 for f in sorted((w/'figures').glob('*.tif')):
  with Image.open(f) as im:assert im.n_frames==1 and im.size==(7016,4961) and im.info['dpi']==(600.,600.)
  proofs.append(dict(file=f.name,SHA256=sha(f),bytes=f.stat().st_size))
 assert len(proofs)==4;js(w/'figure_reproduction_verification.json',proofs);print('FIGURES_PASS: 4 one-page 600 dpi TIFFs',flush=True)

def confidence(w):
 assert (w/'catalog.json').exists(),'Run render first'
 ids=list(registry());call(SUPPORT/'infer_roma.py','--work',w,'--view','posed','--positions',*ids)
 from render_results import confidence_position,csvwrite
 area=[];scores=[]
 for pos in ids:
  m=json.loads((w/'deliverable'/pos/'geometry_metadata.json').read_text());area.extend(dict(position_id=pos,tube_id=tube_id(pos),**{k:v for k,v in row.items() if k!='position'}) for row in m['area_rows'])
  scores.extend(dict(position_id=pos,tube_id=tube_id(pos),**{k:v for k,v in row.items() if k!='position'}) for row in confidence_position(w,m,force=True))
 (w/'tables').mkdir(exist_ok=True);csvwrite(w/'tables/common_area_ratios.csv',area);csvwrite(w/'tables/cumulative_confidence.csv',scores)
 print('CONFIDENCE_PASS: full RoMa fields and cumulative QC recomputed; inspect actual device/provenance',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render','figures','confidence','all']);p.add_argument('--work',type=Path,required=True);p.add_argument('--raw-root',type=Path,help='Optional existing originals: <position>/T1.ext; legacy corrected-site name accepted only as source alias');a=p.parse_args();w=a.work.resolve();w.mkdir(parents=True,exist_ok=True)
 for stage in (['prepare','render','confidence','figures'] if a.stage=='all' else [a.stage]):
  if stage=='prepare':prepare(w,a.raw_root)
  elif stage=='render':render(w)
  elif stage=='figures':figures(w)
  else:confidence(w)
if __name__=='__main__':main()
