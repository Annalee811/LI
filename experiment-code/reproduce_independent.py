"""Replay raw class-head predictions from bundled fixed adapter features, without GT.

This is a portable entry point for this conditional study, not a raw-RGB deployment
replacement. Features retain the documented approximation of the frozen adapter.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import torch
from PIL import Image
from study_model import Network

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs-root',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variant',default='all',choices=['all','full','no_lstm','no_gat','no_gates','current_only'])
    ap.add_argument('--seed',type=int)
    ap.add_argument('--position')
    a=ap.parse_args(); torch.set_num_threads(1)
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Choose a new empty output directory to preserve results.')
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((a.model_root/'reports/locked_protocol.json').read_text())
    norm=dict(np.load(a.model_root/'data/train_only_normalization.npz'))
    positions=sorted(p for p in a.inputs_root.iterdir() if p.is_dir() and (p/'graph.npz').exists())
    if a.position: positions=[p for p in positions if p.name==a.position]
    assert positions, 'No prepared positions found'
    variants=protocol['models'] if a.variant=='all' else [a.variant]
    seeds=protocol['seeds'] if a.seed is None else [a.seed]
    for variant in variants:
        for seed in seeds:
            ckpath=a.model_root/'experiments'/variant/str(seed)/'best.pt'
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            net=Network(variant); net.load_state_dict(ck['backbone']); net.eval()
            for src in positions:
                meta=json.loads((src/'adapter_provenance.json').read_text())
                g=dict(np.load(src/'graph.npz'))
                extra=np.load(src/'extra.npz')['geometry']
                frames=np.load(src/'frames.npz')['frames'].astype(np.float32)
                d={k:torch.from_numpy(g[k]) for k in ['pairs','attrs','phase','centers']}
                d.update(x=torch.from_numpy(((g['x']-norm['x_mean'])/norm['x_std']).astype(np.float32)),
                    extra=torch.from_numpy(((extra-norm['geometry_mean'])/norm['geometry_std']).astype(np.float32)),
                    frames=torch.from_numpy(frames), part=torch.zeros(len(g['x']),dtype=torch.int64),
                    partition_frame_masks=torch.ones((1,*frames.shape[-2:])),
                    native_width=meta['native_width'],native_height=meta['native_height'])
                dest=a.output/variant/str(seed)/src.name;dest.mkdir(parents=True)
                with torch.inference_mode(): h,_,_=net(d,external=True)
                for ti,t in enumerate(meta['periods']):
                    q=dict(np.load(src/(t+'.npz'))); result=np.zeros(tuple(q['shape']),np.uint8)
                    if ti==0: result.ravel()[q['pixel']]=4
                    else:
                        with torch.inference_mode():
                            for i in range(0,len(q['pixel']),32768):
                                sl=slice(i,i+32768)
                                values=net.pixel_logits(h,torch.from_numpy(q['node'][sl]),torch.from_numpy(q['pf'][sl])).argmax(1).numpy()+1
                                result.ravel()[q['pixel'][sl]]=values.astype(np.uint8)
                    Image.fromarray(result).save(dest/(t+'_classes.png'))
                (dest/'inference_status.json').write_text(json.dumps(dict(
                    status='prediction_complete_GT_pending',variant=variant,seed=seed,
                    checkpoint_SHA256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
                    position=src.name,periods=meta['periods'],GT_read_at_inference=False,
                    accuracy=None,adapter='Frozen prepared features, documented transfer approximation'),indent=2))
                print('REPLAYED',variant,seed,src.name,flush=True)
if __name__=='__main__': main()
