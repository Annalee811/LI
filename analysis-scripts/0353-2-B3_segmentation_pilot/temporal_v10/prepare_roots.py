"""Reuse each frozen v9 root model's own masks; no manual support at inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--roots',type=Path,required=True);args=a.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_v6'))
    from continuity_v8 import proposals
    from evaluate_three_v9 import result
    records=[]
    for name in c.MODELS:
        base=args.roots/name;sel=json.loads((base/'postprocessing_selection.json').read_text());classifier=json.loads((base/'link_classifier.json').read_text())
        for col,left in enumerate(c.COLS):
            out=args.data/'root_gates'/name/f'C{col}';out.mkdir(parents=True,exist_ok=True)
            folder=base/'probabilities'/('val' if col==3 else 'test/C4' if col==4 else f'train/C{col}')
            for t in c.PERIODS:
                src=folder/f'{t}_probability.npy';prob=np.load(src)
                cached=base/'evaluation'/('val' if col==3 else 'test')/f'{t}_after.png'
                if col in [3,4] and cached.exists():
                    gate=np.array(Image.open(base/'evaluation'/('val' if col==3 else 'test')/f'{t}_after.png'))>0
                    mask_source='Exact previously exported v9 mask'
                else:
                    rgb=np.array(Image.open(args.data/'upper'/f'{t}_RGB.png'))[:,left:left+448]
                    _,cc=proposals(prob,rgb,sel['threshold']);gate,_,_,_=result(prob,rgb,sel['threshold'],cc,sel['configuration'],classifier);mask_source='Same frozen v9 postprocessing'
                f=out/f'{t}.png';Image.fromarray(gate.astype(np.uint8)*255).save(f)
                records.append(dict(model=name,column=col,period=t,probability_sha256=c.sha(src),gate_sha256=c.sha(f),root_pixels=int(gate.sum()),source=mask_source))
            print('FROZEN_ROOT_GATES',name,col,flush=True)
    c.js(args.data/'root_gate_provenance.json',dict(human_labels_used_for_gate=False,heads_receive_root_only_RGB=True,records=records))
if __name__=='__main__':main()
