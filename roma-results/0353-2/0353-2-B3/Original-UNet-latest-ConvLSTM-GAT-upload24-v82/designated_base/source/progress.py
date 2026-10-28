"""Read saved progress only; never infer completed epochs from wall time."""
import argparse, csv, json, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    result=dict(run=str(a.run),GAT=False,reference='root_annotation_progress (18).json',models={})
    for m in ['unet','deeplab','segformer']:
        d=a.run/m;rec={}
        if (d/'status.json').exists():rec.update(json.loads((d/'status.json').read_text()))
        if (d/'history.csv').exists():
            with (d/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
            rec['last_saved_epoch']=int(history[-1]['epoch'])
            completed=[r for r in history if int(r['epoch'])>0]
            if len(completed)>=3:
                mean=sum(float(r['seconds']) for r in completed[-5:])/len(completed[-5:])
                rec['rough_remaining_training_minutes']=(100-rec['last_saved_epoch'])*mean/60
            rec['current_validation_macro_Dice']=float(history[-1]['validation_macro_Dice'])
        if (d/'completion.json').exists():rec['completion']=json.loads((d/'completion.json').read_text())
        if not rec:rec={'stage':'initial_baseline_evaluation' if (d/'initial_head.pt').exists() else 'preparing'}
        result['models'][m]=rec
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
