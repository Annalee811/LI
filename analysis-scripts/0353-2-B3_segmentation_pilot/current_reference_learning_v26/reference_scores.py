"""Score the three retained final models against the current reference only."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import common as c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();data=c.Data(args.run);records=[];models=[]
    expected={'unet':.884597820539224,'deeplab':.733292673082615,'segformer':.7255330756354242}
    for model in c.MODELS:
        folder=args.source/model;selection=c.read(folder/'selection.json');threshold=float(selection['threshold'])
        ck=folder/('best_weights_fp16.npz' if model=='deeplab' else 'best.pt')
        assert c.sha(ck)==selection['checkpoint_SHA256']
        models.append(dict(model=model,checkpoint=str(ck),checkpoint_SHA256=c.sha(ck),threshold=threshold,selected_epoch=selection['selected_epoch']))
        for split in ['val','test']:
            rows=[r for r in data.rows if r['split']==split];counts=np.zeros(4,np.int64)
            for row in rows:
                _,target=data.load(row)
                p=np.load(folder/'probabilities'/split/(row['image_id']+'_probability.npy'))
                assert p.shape==target.shape and np.isfinite(p).all()
                counts+=c.counts(target,p>=threshold)
            met=c.metrics(counts);met['Accuracy']=float((counts[0]+counts[3])/counts.sum())
            if split=='test':assert abs(met['Dice']-expected[model])<1e-12
            records.append(dict(model=model,name=c.NAMES[model],split=split,reference='current',threshold=threshold,views=len(rows),coordinates=len({r['tile_id'] for r in rows}),**met))
    c.csvout(args.run/'retained_models_current_reference_scores.csv',records)
    c.js(args.run/'reported_models_manifest.json',models)
    print('RETAINED_FINAL_MODEL_CURRENT_REFERENCE_SCORES_VERIFIED',flush=True)
    for row in records:print(row['model'],row['split'],row['Dice'],flush=True)


if __name__=='__main__':main()
