"""New U-Net temporal head trained on24; original binary weights never optimized."""
import json,shutil,sys
from pathlib import Path
import common as c
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_UNet_temporal_upload24_v78_20261028';BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028';LABEL=ROOT/'results/B3_UNet_junction_ConvLSTM_GAT_upload24_v76_20261028';UPLOAD=Path('/workspace/attachments/c64a3bc7-6dc9-4dbd-a57b-166f682547e9/root_annotation_progress (24).json')
def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position('0353-2-B3')=='0353-2-B3';assert not OUT.exists();assert c.sha(UPLOAD)=='cb5a9326b73003a81b44961300de88806378a771514e886faff168a715bf2c8e'
 OUT.mkdir();(OUT/'data/current').mkdir(parents=True);(OUT/'data/upper').mkdir();(OUT/'unet').mkdir();(OUT/'feature_cache').mkdir();immutable=[]
 def keep(p):immutable.append(dict(path=str(p),sha256=c.sha(p)))
 shutil.copy2(UPLOAD,OUT/'data/active_progress.json');keep(OUT/'data/active_progress.json')
 for t in c.PERIODS:
  for src,dest in [(BASE/'data/upper'/f'{t}_RGB.png',OUT/'data/upper'/f'{t}_RGB.png'),(LABEL/'data/reference'/f'{t}_classes.png',OUT/'data/current'/f'{t}_classes.png')]:shutil.copy2(src,dest);keep(dest)
 for n in ['paired_manifest.csv','spatial_manifest.csv']:shutil.copy2(BASE/'data'/n,OUT/'data'/n);keep(OUT/'data'/n)
 (OUT/'feature_cache/unet').symlink_to((BASE/'feature_cache/unet').resolve(),target_is_directory=True)
 p=json.loads((BASE/'preparation.json').read_text());p.update(run=str(OUT),seed=2026102808,uploaded_reference_SHA256=c.sha(UPLOAD),immutable_inputs=immutable,source_upload_has_root_number_annotations=True,features='Exact reused25-channel prediction-only RGB/binary cache; no temporal probability or prior temporal head weight is input',source_upload='root_annotation_progress (24).json',models=['unet'],restart_scope='Randomly initialized U-Net temporal CNN head; frozen binary weights, masks and widths/lengths',subsequent_stage='Freeze this newly trained temporal result; train local-only ConvLSTM/GAT corrections for junctions and short class islands')
 for t in c.PERIODS[1:]:keep(BASE/'feature_cache/unet'/(t+'.npy'))
 roots=Path(p['roots']);keep(roots/'unet/best.pt')
 for t in c.PERIODS:keep(roots/'unet/upper'/f'{t}_root_mask.png');keep(roots/'unet/upper'/f'{t}_probability.npy')
 p['immutable_inputs']=immutable;c.js(OUT/'preparation.json',p);shutil.copytree(HERE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__'));c.js(OUT/'pipeline_status.json',dict(stage='ready',source='24',epochs=100));print(OUT,flush=True)
if __name__=='__main__':main()
