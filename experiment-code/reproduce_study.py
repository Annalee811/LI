"""Reproduce the conditional feature/module study in an empty output folder.
Only the historical baseline-image path is made relative; model/training math are unchanged.
"""
from pathlib import Path
import argparse,shutil,sys,subprocess,json
H=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
 if out.exists()and any(out.iterdir()):raise ValueError('Choose a new empty output directory; existing results are preserved.')
 out.mkdir(parents=True,exist_ok=True);shutil.copytree(H/'data',out/'data',dirs_exist_ok=True);shutil.copytree(H/'code',out/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','decoder_core'));(out/'reports').mkdir(exist_ok=True)
 for name in ['data_audit.json','partition_isolation_check.json','frozen_references.json','independent_protocol.json','annotation_checklist.csv']:shutil.copy2(H/'reports'/name,out/'reports'/name)
 p=out/'code/train_study.py';s=p.read_text();old="Path('/workspace/work/V90_all_microtubes_20261118/model_source/assets/base/unet/full')";assert old in s;s=s.replace(old,"W/'data/baseline_prior'");p.write_text(s)
 (out/'reproduction_path_change.json').write_text(json.dumps(dict(change='Baseline class-map source is changed from historical absolute path to bundled data/baseline_prior',model_math_changed=False,training_settings_changed=False),indent=2));subprocess.run([sys.executable,str(out/'code/run_study.py')],check=True);subprocess.run([sys.executable,str(out/'code/summarize_results.py')],check=True)
if __name__=='__main__':main()
