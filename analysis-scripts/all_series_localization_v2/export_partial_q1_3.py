"""Retain supported T2/T3 observations without inventing a T1 connection."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from assemble_q1 import largest_rect
from render_results import grid,proj,sample,FONT,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1);pos='Q1-3-Z3';base=a.work/'deliverable'/pos
    main_meta=json.loads((base/'geometry_metadata.json').read_text())
    if main_meta['regions']:return
    paths={t:next((a.work/'inputs/raw'/pos).glob(f'T{t}.*')) for t in [2,3]};rgb={t:np.array(Image.open(p).convert('RGB')) for t,p in paths.items()};wh={t:r.shape[1::-1] for t,r in rgb.items()}
    search=json.loads((a.work/'q1_geometry'/pos/'descriptor_search.json').read_text());hyp=search['combined_hypotheses']['T2_to_T3']
    if not hyp:return
    H=np.diag([wh[3][0]/1600]*2+[1])@np.array(hyp[0]['matrix_analysis1600'])@np.diag([1600/wh[2][0]]*2+[1]);q=grid([0,0,*wh[2]]);target=proj(q,H);valid=(target[...,0]>=0)&(target[...,0]<=wh[3][0]-1)&(target[...,1]>=0)&(target[...,1]<=wh[3][1]-1);box=largest_rect(valid);x0,y0,x1,y1=box;q=q[y0:y1,x0:x1];target=target[y0:y1,x0:x1]
    dest=base/'partial_T2_T3_only';dest.mkdir(exist_ok=True)
    aa=rgb[2][y0:y1,x0:x1];bb=cv2.remap(rgb[3],target[...,0],target[...,1],cv2.INTER_LINEAR);Image.fromarray(aa).save(dest/'T2_partial_RGB.png',dpi=(600,600));Image.fromarray(bb).save(dest/'T3_partial_RGB.png',dpi=(600,600))
    with np.load(a.work/'roma_evidence'/pos/'T2_to_T3/dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
    c=np.minimum(sample(d['source_confidence'],q,wh[2]),sample(d['target_confidence'],target,wh[3]));f=(sample(d['forward'],q,wh[2])+1)*[wh[3][0]/2,wh[3][1]/2]-.5;b=(sample(d['backward'],f,wh[3])+1)*[wh[2][0]/2,wh[2][1]/2]-.5;effective=(c>=.5)&(np.linalg.norm(f-target,axis=-1)*wh[2][0]/wh[3][0]<=15)&(np.linalg.norm(b-q,axis=-1)<=10)
    Image.fromarray(np.rint(c*65535).astype('uint16')).save(dest/'T2_to_T3_confidence_16bit.png');Image.fromarray(effective.astype('uint8')*255).save(dest/'T2_to_T3_effective_mask.png');Image.fromarray(np.full(c.shape,255,np.uint8)).save(dest/'T2_to_T3_geometric_mask.png')
    canvas=Image.new('RGB',(aa.shape[1]*2,aa.shape[0]+100),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,27)
    for i,(t,im) in enumerate([(2,aa),(3,bb)]):canvas.paste(Image.fromarray(im),(i*aa.shape[1],100));draw.text((i*aa.shape[1]+10,10),f'Q1-3-Z3 | T{t}',font=font,fill='#263947');draw.text((i*aa.shape[1]+10,50),'仅 T2/T3 局部对应；尚未连接 T1',font=font,fill='#A66C32')
    canvas.save(dest/'T2_T3_partial_comparison.png',dpi=(600,600))
    np.testing.assert_array_equal(np.array(Image.open(dest/'T2_partial_RGB.png')),aa);np.testing.assert_array_equal(np.array(Image.open(dest/'T3_partial_RGB.png')),bb)
    (dest/'partial_metadata.json').write_text(json.dumps(dict(position=pos,status='partial_T2_T3_only_not_a_T1_referenced_sequence',reference_period='T2',T2_box_xyxy=box,T2_to_T3_native_matrix=H.tolist(),automatic_evidence=hyp[0],source_sha256={f'T{t}':sha(p) for t,p in paths.items()},validation=dict(T2_native_crop_exact=True,T3_single_similarity_exact=True),note='Excluded from full-sequence common-area statistics. T1 could not be linked with sufficient evidence.'),ensure_ascii=False,indent=2))
    print('PARTIAL_Q1_3_SAVED',box,'T2 reference only',flush=True)
if __name__=='__main__':main()
