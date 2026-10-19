"""Re-localize corrected Q1 images with multi-mode RootSIFT and RoMa evidence.

Only global similarities are eligible; alternative observation bands stay separate.
Automatic descriptor residuals are not manual landmark errors.
"""
import argparse,itertools,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image

def project(p,H):return p@H[:2,:2].T+H[:2,2]

def modes(pa,pb,width=1600,minimum=12):
    left=np.arange(len(pa));out=[]
    for _ in range(6):
        if len(left)<minimum:break
        H,k=cv2.estimateAffinePartial2D(pa[left].astype('float32'),pb[left].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=20000,confidence=.999,refineIters=20)
        if H is None:break
        ii=left[k.ravel()>0];left=np.setdiff1d(left,ii)
        H=np.vstack([H,[0,0,1]]);scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]))
        if len(ii)<minimum or not (.8<scale<1.2 and abs(angle)<10):continue
        errors=np.linalg.norm(project(pa[ii],H)-pb[ii],axis=1);span=np.ptp(pa[ii],axis=0)
        # Conditional four spatial folds: refit training anchors, measure held-out anchors.
        folds=(np.floor(pa[ii,0]/100)+3*np.floor(pa[ii,1]/100)).astype(int)%4;cv=[]
        for fold in range(4):
            train=folds!=fold;test=~train
            if train.sum()<6 or not test.any():continue
            f,_=cv2.estimateAffinePartial2D(pa[ii][train].astype('float32'),pb[ii][train].astype('float32'),method=cv2.LMEDS)
            if f is not None:cv.extend(np.linalg.norm(project(pa[ii][test],np.vstack([f,[0,0,1]]))-pb[ii][test],axis=1).tolist())
        out.append(dict(matrix_analysis1600=H.tolist(),anchors=len(ii),source_span_xy=span.tolist(),residual_median_analysis1600=float(np.median(errors)),conditional_spatial_CV_median_analysis1600=float(np.median(cv)) if cv else None,region_evidence_pass=bool(len(ii)>=20 and span[0]>=400 and cv and np.median(cv)<=8)))
    return out

def descriptor_search(folder,out):
    images={int(p.stem[1:]):p for p in folder.iterdir() if p.is_file()};small={t:cv2.resize(np.array(Image.open(p).convert('RGB')),(1600,round(Image.open(p).height*1600/Image.open(p).width))) for t,p in images.items()}
    rows={f'T{a}_to_T{b}':[] for a,b in itertools.combinations(sorted(images),2)};points={k:[] for k in rows}
    for mode in ['clahe','blur','ridge']:
        desc={}
        for t,im in small.items():
            g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            if mode=='blur':g=cv2.GaussianBlur(g,(0,0),1.6)
            g=cv2.createCLAHE(2,(12,12)).apply(g)
            if mode=='ridge':g=cv2.createCLAHE(2,(12,12)).apply(np.maximum.reduce([cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))) for n in [9,15,23]]))
            k,d=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.0035,edgeThreshold=20).detectAndCompute(g,None)
            d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-8)) if d is not None else np.empty((0,128),np.float32);desc[t]=(k,d)
        for a,b in itertools.combinations(sorted(images),2):
            ka,da=desc[a];kb,db=desc[b];key=f'T{a}_to_T{b}';matches=[]
            if len(da)>=2 and len(db)>=2:
                bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.84*n.distance};matches=[m for m,n in ab if m.distance<.84*n.distance and rev.get(m.trainIdx)==m.queryIdx]
            pa=np.array([ka[m.queryIdx].pt for m in matches],np.float32).reshape(-1,2);pb=np.array([kb[m.trainIdx].pt for m in matches],np.float32).reshape(-1,2)
            rows[key].append(dict(mode=mode,mutual_matches=len(matches),hypotheses=modes(pa,pb)))
            points[key].append(np.concatenate([pa,pb],1))
            print('SIFT',folder.name,key,mode,len(matches),[(v['anchors'],round(v['matrix_analysis1600'][1][2])) for v in rows[key][-1]['hypotheses']],flush=True)
    out.mkdir(parents=True,exist_ok=True)
    combined={}
    for key,ps in points.items():
        pp=np.concatenate(ps);_,ix=np.unique(np.rint(pp*2).astype(int),axis=0,return_index=True);pp=pp[np.sort(ix)]
        np.savez_compressed(out/(key+'_descriptor_matches.npz'),source_xy_analysis1600=pp[:,:2],target_xy_analysis1600=pp[:,2:]);combined[key]=modes(pp[:,:2],pp[:,2:])
    (out/'descriptor_search.json').write_text(json.dumps(dict(position=folder.name,per_mode=rows,combined_hypotheses=combined,validation='Conditional descriptor cross-validation, not manual TRE.'),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--positions',nargs='*');a=p.parse_args();cv2.setNumThreads(1);cv2.setRNGSeed(20261019)
    for folder in sorted((a.work/'inputs/raw').glob('Q1*')):
        if a.positions and folder.name not in a.positions:continue
        out=a.work/'q1_geometry'/folder.name
        if (out/'descriptor_search.json').exists():continue
        descriptor_search(folder,out)

if __name__=='__main__':main()
