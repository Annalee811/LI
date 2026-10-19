"""Process completed RoMa positions while remaining inference runs; resumable."""
import argparse,json,time
from pathlib import Path
import cv2
from render_results import confidence_position,aggregate

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args();cv2.setNumThreads(1)
while True:
    for path in sorted((a.work/'deliverable').glob('*/geometry_metadata.json')):
        meta=json.loads(path.read_text());pos=meta['position'];ds=meta['periods']
        if not meta['regions'] or (path.parent/'confidence/confidence_summary.json').exists():continue
        ready=all((a.work/'posed_roma_evidence'/pos/f'T{x}_to_T{y}'/'completed.json').exists() for x,y in zip(ds,ds[1:]))
        if ready:confidence_position(a.work,meta)
    if a.once:break
    if (a.work/'inference_complete.json').exists() and len(list((a.work/'deliverable').glob('*/geometry_metadata.json')))==19:break
    time.sleep(20)
aggregate(a.work)
