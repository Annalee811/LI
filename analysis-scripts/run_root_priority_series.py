#!/usr/bin/env python3
"""Run the established full-RoMa root-prioritized workflow for a selected series."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_0353_1_root_priority as workflow
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--series',required=True)
parser.add_argument('--exclude',nargs='*',default=[])
args=parser.parse_args()
workflow.POSITIONS=sorted(p.name for p in workflow.INPUT.glob(args.series+'-*') if p.is_dir() and p.name not in args.exclude)
if not workflow.POSITIONS:raise SystemExit('No selected positions')
selected=[]
for pos in workflow.POSITIONS:
 periods=[f.stem.upper() for f in (workflow.INPUT/pos).iterdir() if f.stem.upper() in ['T1','T2','T3','T4']]
 if len(periods)<=2:print('SKIP',pos,'only',len(periods),'available periods',flush=True);continue
 if 'T1' not in periods:raise SystemExit(f'{pos}: T1 reference is missing')
 selected.append(pos)
workflow.POSITIONS=selected
if not selected:raise SystemExit('No positions with at least three periods and T1 reference')
workflow.CACHE=workflow.ROOT/f'{args.series}_root_priority_configuration'
workflow.CACHE.mkdir(exist_ok=True)
print('SELECTED POSITIONS',workflow.POSITIONS,'EXCLUDED',args.exclude,flush=True)
workflow.main()
