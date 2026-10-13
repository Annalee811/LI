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
for pos in workflow.POSITIONS:
 periods=[f.stem.upper() for f in (workflow.INPUT/pos).iterdir()]
 if not all(f'T{i}' in periods for i in [1,2,3,4]):raise SystemExit(f'{pos}: this four-period runner needs complete T1–T4 inputs')
workflow.CACHE=workflow.ROOT/f'{args.series}_root_priority_configuration'
workflow.CACHE.mkdir(exist_ok=True)
print('SELECTED POSITIONS',workflow.POSITIONS,'EXCLUDED',args.exclude,flush=True)
workflow.main()
