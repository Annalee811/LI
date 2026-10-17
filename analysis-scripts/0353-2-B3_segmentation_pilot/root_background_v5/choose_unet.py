#!/usr/bin/env python3
"""Choose binary-only U-Net refinement by validation Dice, never C4."""
import argparse,json,shutil
from pathlib import Path
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();record=a.run/'unet_optimization.json'
    if record.exists():print('U_NET_CHOICE_ALREADY_FROZEN',flush=True);return
    items=[]
    for folder in ['unet_stage1_initial','unet_precision_candidate']:
        path=a.run/folder;run=json.loads((path/'training_run.json').read_text())
        metrics_path=path/'validation_metrics.json';score=json.loads(metrics_path.read_text())['root']
        assert c.sha(path/'best.pt')==run['checkpoint_sha256']
        items.append(dict(folder=folder,root=score,configuration=run,checkpoint_sha256=run['checkpoint_sha256']))
    chosen=max(items,key=lambda q:(q['root']['Dice'],q['root']['Recall']))
    if chosen['folder']=='unet_precision_candidate':
        source=a.run/chosen['folder'];dest=a.run/'unet'
        for x in source.iterdir():
            if x.is_file():shutil.copyfile(x,dest/x.name)
        shutil.copytree(source/'predictions/val',dest/'predictions/val',dirs_exist_ok=True)
    c.js(record,dict(selected=chosen['folder'],criterion='Calibrated validation strict root Dice; no C4 prediction/scoring used',
        trials=items,task='Both stages RGB-only root/background; no previous-period or temporal input/loss'))
    print('ROOT_UNET_CHOICE',chosen['folder'],chosen['root'],flush=True)
if __name__=='__main__':main()
