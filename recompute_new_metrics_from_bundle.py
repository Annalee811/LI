"""Recompute all new-only counts from bundled binary predictions and fixed GT.

No multiclass model or original machine paths are required by this entry point.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
from PIL import Image
def counts(y,p):return dict(TP=int((y&p).sum()),FP=int((~y&p).sum()),FN=int((y&~p).sum()))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
 a.output.mkdir(parents=True,exist_ok=True);W=a.data_root;results=[];truth=np.load(W/'New_only_reproduction/heldout_new_truth.npz')
 def from_subset(folder):
  n=dict(TP=0,FP=0,FN=0)
  for t in ['T2','T3','T4']:
   p=np.array(Image.open(folder/(t+'_new_mask.png'))).ravel()[truth[t+'_pixel']]==255;y=truth[t+'_new_GT']
   d=counts(y,p)
   for k in n:n[k]+=d[k]
  return n
 def complete(d):return dict(**d,Precision=d['TP']/(d['TP']+d['FP'])if d['TP']+d['FP']else None,Recall=d['TP']/(d['TP']+d['FN'])if d['TP']+d['FN']else None)
 for csvname,pathkeys in [('New_root_internal_each_run.csv',('internal','variant','seed')),('New_root_paired_decoder.csv',('decoder','seed','stage'))]:
  with(W/'metrics'/csvname).open(encoding='utf-8-sig')as f:rr=list(csv.DictReader(f))
  for r in rr:
   folder=W/'New_only_reproduction'/pathkeys[0]/r[pathkeys[1]]/r[pathkeys[2]];m=complete(from_subset(folder))
   for k in ['TP','FP','FN']:assert m[k]==int(r[k]),(csvname,r,k)
   results.append(dict(protocol=pathkeys[0],id=str(folder.relative_to(W)),**m))
 with(W/'metrics/New_root_B3_full_image_historical_agreement.csv').open(encoding='utf-8-sig')as f:rr=list(csv.DictReader(f))
 for model,label in [('V66_frozen','V66'),('V90_automatic_historical','V90')]:
  n=dict(TP=0,FP=0,FN=0)
  for t in ['T2','T3','T4']:
   gt=np.array(Image.open(W/'B3_human_new_truth'/(t+'_new_GT.png')));p=np.array(Image.open(W/'New_only_reproduction/historical'/label/(t+'_new_mask.png')))==255;v=gt!=255;d=counts(gt[v]==1,p[v])
   for k in n:n[k]+=d[k]
  r=next(r for r in rr if r['model']==model and r['period']=='T2-T4_pooled')
  for k in n:assert n[k]==int(r[k])
  results.append(dict(protocol='full_B3_historical',id=model,**complete(n)))
 (a.output/'Recomputed_new_root_Precision_Recall.json').write_text(json.dumps(dict(passed=True,results=results),indent=2));print('RECOMPUTED_NEW_ONLY',len(results),'evaluations; all counts match archived new-only tables')
if __name__=='__main__':main()
