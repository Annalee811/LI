"""One position only: package verified artifacts and complete standalone scripts."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work;out=r/'deliverable/0353-1-Z2';proof=json.loads((out/'evaluation/verification.json').read_text());assert proof['coarse_T2_T3_not_counted_as_point_success'];source=Path(__file__).resolve().parent;dst=out/'scripts';dst.mkdir(exist_ok=True)
 for f in source.iterdir():
  if f.is_file():shutil.copy2(f,dst/f.name)
 support=source/'support'
 if not support.exists():support=source.parent/'all_series_localization_v2'
 for name in ['render_results.py','infer_roma.py','localize_q1.py','assemble_q1.py','model_provenance.json']:
  target=dst/'support'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(support/name,target)
 shutil.copytree(support/'vendor',dst/'support/vendor',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(source/'README.md',out/'README.md')
 evidence=out/'numeric_evidence';evidence.mkdir(exist_ok=True)
 for name in ['search','tight_gray']:
  folder=r/name/'roma_evidence/0353-1-Z2'
  for f in sorted(folder.glob('*/*')):
   if f.is_file():target=evidence/name/f.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 for f in sorted((r/'posed_roma_evidence/0353-1-Z2').glob('*/*')):
  if f.is_file():target=evidence/'posed'/f.relative_to(r/'posed_roma_evidence/0353-1-Z2');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 shutil.copytree(r/'descriptor_evidence',evidence/'descriptor_evidence',dirs_exist_ok=True)
 for name in ['coarse_bridge.json','coarse_bridge_score.npz','registry.json','user_region_priors.json']:
  shutil.copy2(r/name,evidence/name)
 for f in r.glob('T*_root_points.npz'):shutil.copy2(f,evidence/f.name)
 manifest=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='file_manifest.json'];(out/'file_manifest.json').write_text(json.dumps(manifest,indent=2));archive=r/'0353-1-Z2_root_guided_localization_v3.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,'0353-1-Z2_root_guided_localization_v3/'+str(f.relative_to(out)))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None;count=len(z.namelist())
 result=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=count,CRC_test_passed=True,position='0353-1-Z2',four_period_crop_status='candidate observation ROI; T2/T3 coarse link unverified');(r/'archive_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
