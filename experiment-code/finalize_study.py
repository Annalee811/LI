from pathlib import Path
import subprocess,json,time,sys,hashlib,shutil,zipfile
W=Path(__file__).resolve().parents[1]
for _ in range(720):
 progress=json.loads((W/'reports/run_progress.json').read_text())if(W/'reports/run_progress.json').exists()else[]
 if any(r['status']=='failed'for r in progress):raise RuntimeError('Training failed; inspect logs')
 if all((W/name).exists()for name in ['TRAINING_COMPLETE.json','DECODER_STUDY_COMPLETE.json','INDEPENDENT_PREDICTIONS_COMPLETE.json']):break
 time.sleep(5)
else:raise RuntimeError('Experiments not ready')
for py,name in [(sys.executable,'final_checks.py'),(sys.executable,'summarize_results.py'),('python','plot_and_tables.py'),('python','write_report.py')]:
 with(W/(name+'.log')).open('w')as f:subprocess.run([py,str(W/'code'/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
# Record original frozen weights and source labels unchanged.
O=Path('/workspace/work/V90_all_microtubes_20261118');cfg=json.loads((O/'delivery/V90_transfer_model/deployment_config.json').read_text());checks={}
for rel,digest in cfg['weights_SHA256'].items():
 p=O/'delivery/V90_transfer_model'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;checks[rel]=digest
(W/'reports/original_weights_preserved.json').write_text(json.dumps(dict(passed=True,weights_SHA256=checks),indent=2));(W/'ALL_EXPERIMENTS_READY.json').write_text(json.dumps(dict(ready=True,retrained_runs=15,epochs_per_run=100,independent_predictions=390,independent_accuracy_available=False),indent=2));print('ALL_EXPERIMENTS_READY',flush=True)
