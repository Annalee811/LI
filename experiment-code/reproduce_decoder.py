"""Replay the paired whole-field decoder experiment in a new output directory.

Only source paths are staged; original numerical/model code is unchanged.
Requires the separate Decoder_inputs.zip and the main study bundle.
"""
from pathlib import Path
import argparse, shutil, sys, importlib.util
import torch
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,choices=[20261123,20261124,20261125])
    a=ap.parse_args(); a.output=a.output.resolve(); a.inputs=a.inputs.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
    (a.output/'code').mkdir(parents=True,exist_ok=True);(a.output/'data').mkdir()
    shutil.copytree(a.model_root/'code',a.output/'code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for n in ['train_only_normalization.npz','junctions.npz']:
        shutil.copy2(a.model_root/'data'/n,a.output/'data'/n)
    seeds=[a.seed] if a.seed is not None else [20261123,20261124,20261125]
    for seed in seeds:
        p=a.output/'experiments/full'/str(seed);p.mkdir(parents=True)
        shutil.copy2(a.model_root/'experiments/full'/str(seed)/'best.pt',p/'best.pt')
    p=a.output/'code/decoder_study.py';s=p.read_text()
    s=s.replace("A=OLD/'model_source/assets'","A=Path("+repr(str(a.inputs))+')')
    s=s.replace("for f in(OLD/'deployment/portable_core').glob('*.py'):","for f in CORE.glob('*.py'):")
    s=s.replace("'/workspace/LI/analysis-scripts/dataset_identity.py'",repr(str(a.inputs/'analysis-scripts/dataset_identity.py')))
    s=s.replace("'/workspace/LI/dataset/current'",repr(str(a.inputs/'dataset/current')))
    p.write_text(s);sys.path.insert(0,str(a.output/'code'))
    spec=importlib.util.spec_from_file_location('paired_decoder_replay',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for seed in seeds:module.one(seed)
    print('PAIRED_DECODER_REPLAY_COMPLETE',len(seeds))
if __name__=='__main__':main()
