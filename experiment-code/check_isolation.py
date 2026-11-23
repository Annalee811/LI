from pathlib import Path
import torch,json
from train_study import dataset
from study_model import initialize
W=Path(__file__).resolve().parents[1];torch.set_num_threads(1);g,d,*_=dataset();report={}
for variant in ['full','no_lstm','no_gat','no_gates','current_only']:
 net,_=initialize(variant,20261123);net.eval();perturbed={k:v.clone()for k,v in d.items()};perturbed['x'][d['part']!=0]+=7;perturbed['extra'][d['part']!=0]-=7
 # Disturb low-resolution non-training spatial cells; partition masks must shield train context.
 other=d['partition_frame_masks'][0]==0;perturbed['frames'][:,:,other]=0
 with torch.inference_mode():h,_,_=net(d);q,_,_=net(perturbed)
 err=float((h[d['part']==0]-q[d['part']==0]).abs().max());assert err==0,(variant,err);report[variant]=dict(train_embedding_max_difference_after_other_partition_perturbation=err,passed=True)
(W/'reports/partition_isolation_check.json').write_text(json.dumps(report,indent=2));print('PARTITION_ISOLATION_PASSED',len(report))
