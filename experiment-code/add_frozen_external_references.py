from pathlib import Path
import csv,json,hashlib,shutil
W=Path(__file__).resolve().parents[1];O=Path('/workspace/work/V90_all_microtubes_20261118');rows=list(csv.DictReader((W/'reports/annotation_checklist.csv').open(encoding='utf-8-sig')));written=[]
for model in ['V66_pairwise_frozen','V90_transfer_reference']:
 for r in rows:
  pos=r['position_id'];t=r['period'];src=O/'transfer_candidates'/pos/'base/unet/full'/(t+'_classes.png')if model=='V66_pairwise_frozen'else O/'delivery/V90_transfer_candidates'/pos/(t+'_classes.png');dst=W/'independent_predictions'/model/'frozen'/pos/(t+'_classes.png');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);written.append(dict(model=model,position=pos,period=t,SHA256=hashlib.sha256(src.read_bytes()).hexdigest(),source=str(src),status='Existing frozen automatic prediction reused; no external GT scores'))
(W/'reports/frozen_external_reference_index.json').write_text(json.dumps(written,indent=2));print('FROZEN_REFERENCE_MAPS',len(written))
