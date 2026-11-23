from pathlib import Path
import csv,json,subprocess,sys,torch
from train_study import dataset
from study_model import Network
W=Path(__file__).resolve().parents[1];torch.set_num_threads(1);g,d,p,*_=dataset();net=Network('current_only');ck=torch.load(W/'experiments/current_only/20261123/best.pt',weights_only=False);net.load_state_dict(ck['backbone']);net.eval();changed={k:v.clone()for k,v in d.items()};changed['frames'][[0,1,3]]=0;changed['x'][d['phase']!=2]+=8;changed['x'][:,7]+=11;changed['x'][:,14:]+=13;changed['extra'][:,3]+=17
with torch.inference_mode():
 h,_,_=net(d);q,_,_=net(changed);herror=float((h[d['phase']==2]-q[d['phase']==2]).abs().max());assert herror==0;nn=torch.from_numpy(p['T3']['node'][:100]);pf=torch.from_numpy(p['T3']['pf'][:100]);pf2=pf.clone();pf2[:,:3]+=19;pf2[:,8:13]-=23;err=float((net.pixel_logits(h,nn,pf)-net.pixel_logits(q,nn,pf2)).abs().max());assert err==0
folder=W/'smoke/empty_GT_gate';folder.mkdir(parents=True,exist_ok=True);src=W/'reports/annotation_checklist.csv';rows=list(csv.DictReader(src.open(encoding='utf-8-sig')))
for r in rows:r.update(annotation_state='human_complete',annotator='TEST_ONLY_INVALID_TEMPLATE',annotation_source='Software negative control; NOT real human annotation')
manifest=folder/'invalid_template_manifest.csv'
with manifest.open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=subprocess.run([sys.executable,str(W/'code/evaluate_independent.py'),'--manifest',str(manifest),'--prediction-dir',str(folder/'not_needed'),'--output',str(folder/'status')],capture_output=True,text=True);assert r.returncode==3;status=json.loads((folder/'status/evaluation_status.json').read_text());assert status['metrics']is None and len(status['missing_GT'])==26;assert all('unannotated'in q['reason']for q in status['missing_GT'])
report=dict(passed=True,trained_current_frame_classifier_history_perturbation_max_logit_difference=err,train_test_frame_isolation_already_verified=True,unannotated_GT_templates_rejected=True,no_fake_independent_scores_produced=True);(W/'reports/baseline_and_GT_gate_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
