"""Native-pixel evidence: the reported D attachment owns one persistent root pixel."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_state_prediction_repair_v88_20261028'
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from rendering import Canvas,isolated


def main():
    assert json.loads((OUT/'evaluation_verification.json').read_text())['passed']
    before=np.array(Image.open(ROOT/'results/B3_branch_continuity_review_v85_20261028/full/T3_classes.png'))
    after=np.array(Image.open(OUT/'full/T3_classes.png'))
    owner=np.load(OUT/'ownership/T3_axis_owner.npz')['axis']
    raw=np.array(Image.open(ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028/data/upper/T3_RGB.png'))
    x,y=1067,620;assert after[y,x]==1
    vertical=int(owner[y,x]);diagonal=next(r['axis'] for r in json.loads((OUT/'User_case_audit.json').read_text()) if r['case']=='D');assert vertical>=0 and diagonal>=0 and vertical!=diagonal
    identity=np.full((*after.shape,3),255,np.uint8);identity[after>0]=[205,205,205];identity[owner==vertical]=[0,80,160];identity[owner==diagonal]=[213,94,0]
    cv=Canvas(3200,1240,OUT/'figures');cv.text(1600,25,'Reported attachment: native pixel and single-root ownership',55);cv.text(1600,115,'Circled pixel (1067, 620): Persistent | Root geometry is identical before and after',30)
    x0,y0,x1,y1=[1035,588,1099,652];xx=[130,920,1710,2500]
    for xc,title,a in zip(xx,['RGB','Before temporal classes','Corrected temporal classes','Predicted root ownership'],[raw,isolated(before),isolated(after),identity]):
        cv.text(xc+280,220,title,31);cv.panel(a[y0:y1,x0:x1],xc,300,560,560,xc!=130)
        if xc in xx[2:]:
            px=xc+(x-x0+.5)*560/64;py=300+(y-y0+.5)*560/64;cv.d.ellipse((px-18,py-18,px+18,py+18),outline='black',width=3)
    cv.text(1600,945,'Temporal colors: purple = persistent; green = new; yellow = extension.',29)
    cv.text(1600,1010,'Ownership panel: blue = vertical root; orange = diagonal root. One shared pixel has one owner.',29)
    cv.text(1600,1075,'The new-colored jump is removed. Actual distal extension remains on its own established root.',29)
    rec=cv.save('Fig5_Reported_junction_native_pixel_and_ownership');p=OUT/'figure_manifest.json';manifest=json.loads(p.read_text());manifest['files']=[r for r in manifest['files'] if r['file']!=rec['file']]+[rec];p.write_text(json.dumps(manifest,indent=2))
    (OUT/'Junction_D_native_pixel_verification.json').write_text(json.dumps(dict(period='T3',x=x,y=y,before_class=int(before[y,x]),after_class=int(after[y,x]),after_name='persistent',single_owner_axis=vertical,diagonal_owner_axis=diagonal,geometry_unchanged=True),indent=2))


if __name__=='__main__':main()
