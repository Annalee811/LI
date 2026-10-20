"""Select an inherited chain penalty using revised validation ONLY."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported();all_reports={}
for name in c.MODELS:
 start=time.monotonic();folder=a.run/name;head=folder/'best_temporal_head.pt';backup=folder/'pre_calibration_temporal_head.pt';assert not backup.exists(),'Use a new version rather than overwrite calibration history';shutil.copy2(head,backup);model=c.load_head(head,device);data=c.Data(a.run,a.roots,name);records=[];best=-float('inf');penalty=None
 for strength in [0.,.5,1.,2.,4.,8.,12.]:
  model.chain_penalty=strength;report=c.evaluate(model,data,'val',device,amp=amp);score=report['revised']['end_to_end']['macro_Dice'];records.append(dict(chain_penalty=strength,split='val',selection_reference='revised',revised_macro_Dice=score,original_macro_Dice=report['original']['end_to_end']['macro_Dice'],revised_class_Dice=[v['Dice'] for v in report['revised']['end_to_end']['classes']]))
  if score>best+1e-6 or abs(score-best)<=1e-6 and (penalty is None or strength>penalty):best=score;penalty=strength
 ck=torch.load(backup,map_location='cpu',weights_only=True);ck['chain_penalty']=penalty;ck['config']['chain_penalty']=penalty;ck['config']['chain_penalty_selected_this_round']=True;ck['config']['decoder_selection']='Fixed0,.5,1,2,4,8,12 grid, revised validation macroDice only; test not read for selection';torch.save(ck,head);model=c.load_head(head,device)
 reports={s:c.evaluate(model,data,s,device,folder/'evaluation'/s,amp=amp) for s in ['val','test']};c.js(folder/'metrics.json',reports);config=json.loads((folder/'config.json').read_text());config.update(chain_penalty=penalty,chain_penalty_selected_this_round=True,decoder_selection=ck['config']['decoder_selection']);c.js(folder/'config.json',config)
 train=json.loads((folder/'training_run.json').read_text());train.update(chain_penalty=penalty,chain_penalty_selected_this_round=True,selected_head_SHA256=c.sha(head),selected_revised_validation_macro_Dice=best,decoder_calibration_seconds=time.monotonic()-start);c.js(folder/'training_run.json',train);c.js(folder/'decoder_selection.json',dict(candidates=records,selected_chain_penalty=penalty,selected_revised_validation_macro_Dice=best,selected_on='15 validation pairs only',test_used_for_selection=False,pre_calibration_head_SHA256=c.sha(backup),head_SHA256=c.sha(head)))
 all_reports[name]=reports;c.js(a.run/'all_metrics.json',all_reports);print('DECODER_CALIBRATED',name,'penalty',penalty,'validation_macro',best,'test_macro',reports['test']['revised']['end_to_end']['macro_Dice'],flush=True)
