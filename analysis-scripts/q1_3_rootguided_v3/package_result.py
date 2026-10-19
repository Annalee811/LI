"""Create one audited ZIP without touching older results or other locations."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.work/'deliverable';source=Path(__file__).resolve().parent;assert json.loads((root/'evaluation/verification.json').read_text())['T1_unknown_not_zero']
 dst=root/'scripts';dst.mkdir(exist_ok=True)
 for f in source.rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  rel=f.relative_to(source);target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 vendor=source/'vendor'
 if not vendor.exists():vendor=source.parent/'all_series_localization_v2/vendor'
 if vendor.exists():shutil.copytree(vendor,dst/'vendor',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 audit=root/'numeric_evidence';audit.mkdir(exist_ok=True)
 for k in ['T1_to_T2','T1_to_T3','T2_to_T3','enhanced_T1_to_T2','enhanced_T1_to_T3','posed_T2_to_T3']:
  for name in ['dense.npz','matches.npz']:
   f=a.work/k/name
   if f.exists():target=audit/k/name;target.parent.mkdir(exist_ok=True);shutil.copy2(f,target)
 for name in ['search_priors.json','hypotheses.json','root_pose_candidates.json','descriptor_probe.json']:
  shutil.copy2(a.work/name,audit/name)
 manifest=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='file_manifest.json'];(root/'file_manifest.json').write_text(json.dumps(manifest,indent=2))
 archive=a.work/'Q1-3-Z3_root_guided_recheck_v3.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for f in sorted(root.rglob('*')):
   if f.is_file():z.write(f,'Q1-3-Z3_root_guided_recheck_v3/'+str(f.relative_to(root)))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None;count=len(z.namelist())
 report=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=count,CRC_test_passed=True,only_position='Q1-3-Z3');(a.work/'archive_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
