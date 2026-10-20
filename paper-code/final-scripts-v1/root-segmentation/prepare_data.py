"""Validate supplied RGB/masks against the fixed input manifest."""
import argparse,csv,json,os,shutil
from pathlib import Path
import common as c
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve();assert not(a.run/'data').exists(),'Choose a new experiment directory'
    base=Path(__file__).resolve().parent;rows=c.csvread(base/'training_manifest.csv');cfg=c.read(base/'configuration.json');assert c.sha(base/'training_manifest.csv')==cfg['manifest_sha256'];groups={}
    for r in rows:
        assert groups.setdefault(r['tile_id'],r['split'])==r['split']
        for kind in ['RGB','mask']:
            rel=Path(r[kind+'_path']);assert rel.parts[0]=='data';src=a.data_root/Path(*rel.parts[1:]);assert c.sha(src)==r[kind+'_sha256'],str(rel)
    for item in c.read(base/'initialization.json'):assert c.sha(a.initial/item['file'])==item['sha256']
    for r in rows:
        for kind in ['RGB','mask']:
            rel=Path(r[kind+'_path']);src=a.data_root/Path(*rel.parts[1:]);dest=a.run/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    shutil.copy2(base/'training_manifest.csv',a.run/'data/manifest.csv');c.js(a.run/'data/summary.json',dict(manifest_SHA256=cfg['manifest_sha256'],initial_checkpoint_dir=os.path.relpath(a.initial.resolve(),a.run),annotation_source='Supplied human-reviewed masks'))
    if (a.data_root/'upper').is_dir():
        for name,h in c.read(base/'inference_inputs.json').items():
            src=a.data_root/'upper'/name;assert c.sha(src)==h;dest=a.run/'data/upper'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    print('INPUTS_VERIFIED',flush=True)
if __name__=='__main__':main()
