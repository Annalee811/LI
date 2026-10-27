from pathlib import Path
import subprocess,sys,time,os,fcntl,json,shutil
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare import O,js,MODELS

def main():
 O.mkdir(exist_ok=True);lock=(O/'pipeline.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Already running')
 for p in HERE.glob('*.py'):shutil.copy2(p,O/'source'/p.name)
 for name,args in [('prepare',[])]+[(m+'_'+kind,['--model',m,'--kind',kind]) for m in MODELS for kind in ['mlp','gat']]:
  if (O/'STOPPED_BY_USER.json').exists():return
  script=HERE/('prepare.py' if name=='prepare' else 'train.py');js(O/'pipeline_status.json',dict(stage=name,pid=os.getpid(),updated_unix=time.time()))
  with (O/(name+'.log')).open('a') as log:
   p=subprocess.run([sys.executable,'-B',str(script),*args],cwd=O.parents[1],stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:js(O/'pipeline_status.json',dict(stage='failed',failed_stage=name,returncode=p.returncode,pid=os.getpid(),updated_unix=time.time()));raise SystemExit(p.returncode)
 if not (O/'training_complete.json').exists():js(O/'training_complete.json',dict(state='six_new_heads_complete',epochs_each=100,models=MODELS,ablations=['mlp','gat'],finished_unix=time.time()))
 # Recovery completes only missing result stages; it does not rerun a completed
 # training module or replace previously saved human labels.
 for script,marker in [('evaluate.py','evaluation_complete.json'),('viewer.py','Root_ID_comparison.html'),('figures.py','figure_manifest.json'),('ablation_figure.py','ablation_figure_manifest.json'),('report.py','RESULTS.md')]:
  if (O/'STOPPED_BY_USER.json').exists():return
  if (O/marker).exists():continue
  js(O/'pipeline_status.json',dict(stage=script,pid=os.getpid(),updated_unix=time.time()))
  with (O/(script+'.log')).open('a') as log:
   result=subprocess.run([sys.executable,'-B',str(HERE/script)],cwd=O.parents[1],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise SystemExit(result.returncode)
 js(O/'pipeline_status.json',dict(stage='all_local_results_complete',updated_unix=time.time()))
if __name__=='__main__':main()
