"""Report real saved epochs and selected-model results without changing training."""
import json
from prepare import OUT,MODELS
def main():
    report={}
    for m in MODELS:
        report[m]={}
        for kind in ['gat','mlp']:
            d=OUT/(m+'_'+kind);record=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else dict(stage='queued')
            if (d/'completion.json').exists():
                r=json.loads((d/'completion.json').read_text());record.update(selected_epoch=r['selected_epoch'],class_Dice={x['name']:x['Dice'] for x in r['test']['classes']},connection_F1=r['test_connection']['F1'])
            report[m][kind]=record
    print(json.dumps(dict(models=report,phase=json.loads((OUT/'pipeline_status.json').read_text())),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
