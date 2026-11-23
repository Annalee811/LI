from pathlib import Path
import json,subprocess,sys,hashlib,time
from concurrent.futures import ThreadPoolExecutor,as_completed
W=Path(__file__).resolve().parents[1];variants=['full','no_lstm','no_gat','no_gates','current_only'];seeds=[20261123,20261124,20261125];config=dict(version='V90-module-study-v1',models=variants,seeds=seeds,epochs=100,threads_per_run=1,parallel_runs=2,initialization='Fresh matched random initialization for shared parameters; V90 trained weights used only as reference',training_steps_per_epoch=1,training_pixel_samples=8192,learning_rate=3e-4,weight_decay=1e-4,optimizer='AdamW',test_selection_forbidden=True,checkpoint_selection='Validation macro Dice only',scope='Internal B3 feature-backbone module study',external_test_tubes=['0353-3','A017-3','Q1-2'],independent_test_status='PENDING_HUMAN_GT',data_SHA256=hashlib.sha256((W/'reports/data_audit.json').read_bytes()).hexdigest(),code_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [W/'code/study_model.py',W/'code/train_study.py']})
p=W/'reports/locked_protocol.json'
if p.exists():assert json.loads(p.read_text())==config
else:p.write_text(json.dumps(config,indent=2))
(W/'logs').mkdir(exist_ok=True);results=[]
def one(v,s):
 out=W/'experiments'/v/str(s)
 if(out/'status.json').exists()and json.loads((out/'status.json').read_text()).get('stage')=='complete':return dict(variant=v,seed=s,status='complete_existing')
 with(W/'logs'/f'{v}_{s}.log').open('a')as f:subprocess.run([sys.executable,str(W/'code/train_study.py'),'--variant',v,'--seed',str(s),'--epochs','100','--threads','1'],stdout=f,stderr=subprocess.STDOUT,check=True)
 return dict(variant=v,seed=s,status='complete')
with ThreadPoolExecutor(max_workers=2)as ex:
 tasks=[ex.submit(one,v,s)for s in seeds for v in variants]
 for future in as_completed(tasks):
  try:r=future.result()
  except Exception as e:r=dict(status='failed',error=str(e))
  results.append(r);(W/'reports/run_progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
if any(r['status']=='failed'for r in results):raise SystemExit('Some runs failed; see saved logs.')
(W/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(complete=True,runs=len(results),epochs_each=100),indent=2));print('ALL_RUNS_COMPLETE',len(results),flush=True)
