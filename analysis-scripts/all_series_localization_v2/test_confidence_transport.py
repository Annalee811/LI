"""Analytic test: transported gradients, original-vs-joined offsets and seams."""
import json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from render_results import confidence_position

def main():
    root=Path(tempfile.mkdtemp(prefix='LI_confidence_transport_'));pos='synthetic';out=root/'deliverable'/pos;(out/'masks').mkdir(parents=True)
    W=H=64;y,x=np.mgrid[:H,:W].astype('float32');g=np.stack([x,y],-1);forward=(g+[1,0]+.5)/[W/2,H/2]-1;backward=(g-[1,0]+.5)/[W/2,H/2]-1
    mask=np.zeros((H,W),np.uint8);mask[:20]=255;mask[44:]=255;Image.fromarray(mask).save(out/'masks/stitched_geometric_overlap_mask.png')
    boxes={'A':[0,0,64,20],'B':[0,30,64,50]};regions=[dict(id=k,matrices={f'T{t}':np.eye(3).tolist() for t in [1,2,3]}) for k in boxes]
    meta=dict(position=pos,periods=[1,2,3],regions=regions,joined_boxes_T1=boxes,stitched_RGB_wh=[64,64],original_wh={f'T{t}':[64,50] for t in [1,2,3]})
    for a,b in [(1,2),(2,3)]:
        d=root/'posed_roma_evidence'/pos/f'T{a}_to_T{b}';d.mkdir(parents=True);(d/'completed.json').write_text('{}')
        c=np.full((H,W),.8,np.float32) if a==1 else .5+.2*x/W+.1*y/H
        np.savez_compressed(d/'dense_evidence.npz',forward=forward.astype('float32'),backward=backward.astype('float32'),source_confidence=c,target_confidence=np.ones((H,W),np.float32),source_wh=[W,H],target_wh=[W,H])
    confidence_position(root,meta)
    with np.load(out/'confidence/T1_to_T3_cumulative_confidence_float32.npz') as z:c=z['confidence']
    # T1 pixel x is transported to x+1 before evaluating the T2->T3 confidence.
    # B starts at original T1 y=30 but stitched y=44, a deliberately different offset.
    for row in [5,49]:
        expected=.8*(.5+.2*(np.arange(61)+1)/W+.1*row/H)
        np.testing.assert_allclose(c[row,:61],expected,atol=2e-6,rtol=0)
    assert np.all(c[20:44]==0) and np.all(c[:,-2:]==0)
    native=np.array(Image.open(out/'confidence/T1_to_T3_cumulative_confidence_native_T1_16bit.png'))/65535
    np.testing.assert_allclose(native[35,:61],c[49,:61],atol=.5/65535+1e-7,rtol=0)
    assert np.all(native[20:30]==0)
    print('PASS analytic dense-chain transport, non-data seams and native T1 restoration')
if __name__=='__main__':main()
