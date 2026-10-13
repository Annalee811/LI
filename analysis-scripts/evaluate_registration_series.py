#!/usr/bin/env python3
"""Evaluate and package a complete four-period RoMa series after its adjacent fields finish."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,sys,json,subprocess,time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--series',required=True)
parser.add_argument('--exclude',nargs='*',default=[])
parser.add_argument('--wait-for-pairs',type=float,default=0,help='Maximum seconds to wait per position for completed adjacent inference')
args=parser.parse_args()
scripts=Path(__file__).resolve().parent
positions=sorted(p.name for p in (scripts/'input/raw/V-变化').glob(args.series+'-*') if p.is_dir() and p.name not in args.exclude)
if not positions:raise SystemExit('No selected positions')

def run(script,arguments,log_name):
 print('START',script,*arguments,flush=True)
 with (scripts/log_name).open('w') as log:
  subprocess.run([sys.executable,str(scripts/script),*arguments],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,*arguments,flush=True)

for pos in positions:
 deadline=time.monotonic()+args.wait_for_pairs
 markers=[scripts/pos/f'T{i}_to_T{i+1}'/'root_priority_completed.json' for i in range(1,4)]
 while not all(p.exists() for p in markers):
  if time.monotonic()>=deadline:raise SystemExit(f'{pos}: adjacent fields incomplete')
  time.sleep(5)
 manifest=json.loads((scripts/pos/'manifest.json').read_text())
 for p in markers:assert json.loads(p.read_text())['input_hashes']==manifest['input_sha256']
 print('ADJACENT READY',pos,flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(run,'evaluate_adjacent_registration.py',['--source',str(scripts/pos),'--output',str(scripts/f'{pos}_registration_evaluation')],f'{pos}_adjacent_evaluation.log'),
        pool.submit(run,'evaluate_continuous_registration.py',['--position',pos],f'{pos}_continuous_evaluation.log')]
  for job in jobs:job.result()
 run('finalize_continuous_registration.py',['--position',pos],f'{pos}_hdf5_export.log')
run('package_registration_series.py',['--series',args.series,'--exclude',*args.exclude],f'{args.series}_package.log')
print('ALL EVALUATION AND PACKAGING COMPLETE',args.series,flush=True)
