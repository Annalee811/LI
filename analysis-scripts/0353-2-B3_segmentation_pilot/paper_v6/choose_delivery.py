#!/usr/bin/env python3
"""Keep frozen model ranking; choose a near-equal-Dice compact filter on C3 only.

The original strict ranking remains immutable. This delivery choice implements
the user's explicit noise-removal objective, with its small score tradeoff shown.
"""
import argparse,copy,csv,json
from pathlib import Path
import protocol as p
from select_refinement import load,evaluate

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);args=a.parse_args()
 original=json.loads((args.run/'selection.json').read_text());out=copy.deepcopy(original);winner=out['winner'];choice=out['candidates'][winner]
 base=choice['selected'];config=base['configuration'];options=[]
 with (args.run/'candidates'/winner/'validation_cleanup_grid.csv').open(encoding='utf-8-sig') as f:
  for row in csv.DictReader(f):
   if (float(row['threshold'])==config['threshold'] and float(row['hysteresis_delta'])==config['hysteresis_delta'] and
       int(row['compact_area'])>0 and row['admissible']=='True' and float(row['Dice'])>=base['metrics']['root']['Dice']-.001 and
       float(row['Precision'])>=base['metrics']['root']['Precision']):options.append(row)
 if options:
  best=max(options,key=lambda r:(float(r['Precision']),float(r['Dice'])));cfg=dict(threshold=config['threshold'],hysteresis_delta=config['hysteresis_delta'],compact_area=int(best['compact_area']))
  score=evaluate(p.Data(args.data),load(args.run/'candidates'/winner/'val'),cfg)
  choice['selected']=dict(configuration=cfg,metrics=score)
 out['delivery_refinement']=dict(original_ranking_file='selection.json',original_ranking_SHA256=p.sha(args.run/'selection.json'),
  model_and_threshold_unchanged=True,criterion='Among same-threshold/same-hysteresis compact filters with C3 Dice loss<=0.001 and original Recall safeguards, prefer higher C3 Precision',
  original_validation=base,delivered_validation=choice['selected'],uses_C4_scores=False,
  timing='Additional delivery cleanup after initial ranking, selected from the already computed C3 grid. C4 was previously viewed and remains a repeated exploratory benchmark.')
 p.js(args.run/'delivery_selection.json',out);print('DELIVERY_CLEANUP',winner,choice['selected']['configuration'],choice['selected']['metrics']['root'],flush=True)
if __name__=='__main__':main()
