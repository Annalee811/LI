"""Reproduce the two fractions from bundled fixed skeleton weights, NumPy only."""
from pathlib import Path
import argparse,csv,json
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
 a.output.mkdir(parents=True,exist_ok=True)
 with(a.bundle/'V90_two_class_all_66_frames.csv').open(encoding='utf-8-sig')as f:rows=list(csv.DictReader(f))
 out=[]
 for r in rows:
  q=np.load(a.bundle/'skeleton_reproduction'/r['position_id']/(r['period']+'_skeleton.npz'));w=q['weight'];cl=q['original_class'];total=w.sum();new=w[cl==2].sum();existing=w[np.isin(cl,[1,3])].sum();baseline=r['baseline']=='True'
  assert abs(total-float(r['total_skeleton_length_px']))<1e-7
  if not baseline:
   assert abs(100*new/total-float(r['new_pct_raw']))<1e-8 and abs(100*existing/total-float(r['existing_pct_raw']))<1e-8
   assert abs(new+existing-total)<1e-7
  out.append(dict(position_id=r['position_id'],period=r['period'],new_pct=None if baseline else 100*new/total,existing_pct=None if baseline else 100*existing/total))
 with(a.output/'Recomputed_two_class_fractions.csv').open('w',encoding='utf-8-sig',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
 (a.output/'reproduction_check.json').write_text(json.dumps(dict(passed=True,frames=len(out),fractions_match_to_tolerance=1e-8,baseline_rates_missing=True),indent=2));print('REPRODUCED_TWO_CLASS_FRACTIONS',len(out))
if __name__=='__main__':main()
