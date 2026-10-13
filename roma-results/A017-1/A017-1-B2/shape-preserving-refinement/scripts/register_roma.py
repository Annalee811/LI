#!/usr/bin/env python3
"""RoMa RGB registration. Pixel maps use zero-based pixel-centre coordinates.

Run from /workspace/LI with .venv/bin/python. No colour correction, cropping,
EXIF rotation, or target resizing is applied. Neural inference uses RoMa's
internal resolution; the dense predictions are interpolated onto native grids.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TORCH_HOME', '/workspace/.cache/torch')
os.environ.setdefault('MPLCONFIGDIR', '/workspace/.cache/matplotlib')
import cv2
import numpy as np
from PIL import Image
import torch


def grid(h, w):
    y, x = np.mgrid[:h, :w].astype(np.float32)
    return np.stack([x, y], -1)


def pixel_map(normalized, size, canvas):
    """align_corners=False: x_px=(x_norm+1)*W/2-0.5."""
    w, h = size
    ow, oh = canvas
    n = cv2.resize(normalized, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return (n + 1) * np.array([w/2, h/2], np.float32) - .5


def remap(a, m, nearest=False, border=0):
    return cv2.remap(a, m[..., 0], m[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def inside(m, size):
    w, h = size
    return (np.isfinite(m).all(-1) & (m[..., 0] >= 0) &
            (m[..., 0] <= w-1) & (m[..., 1] >= 0) & (m[..., 1] <= h-1))


def jacobian(m):
    dx_y, dx_x = np.gradient(m[..., 0])
    dy_y, dy_x = np.gradient(m[..., 1])
    return dx_x * dy_y - dx_y * dy_x


def project(m, homography):
    h = np.asarray(homography, np.float64)
    x, y = m[..., 0], m[..., 1]
    z = h[2,0]*x + h[2,1]*y + h[2,2]
    return np.stack([(h[0,0]*x+h[0,1]*y+h[0,2])/z,
                     (h[1,0]*x+h[1,1]*y+h[1,2])/z],-1).astype(np.float32)


def rgb(path):
    with Image.open(path) as im:
        if im.mode != 'RGB':
            raise ValueError(f'Input must be original RGB, got {im.mode}: {path}')
        return np.array(im)


def save_rgb(path, a):
    Image.fromarray(a).save(path)


def stat(a, mask):
    v = a[mask]
    return {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95))} if v.size else {'median': None, 'p95': None}


def corr(a, b):
    if len(a) < 10 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def output(out, source, target, backward, confidence, valid, metadata,
           forward=None, fconfidence=None, cycle=None, determinant=None):
    out.mkdir(parents=True, exist_ok=True)
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]
    registered = remap(source, backward)
    registered[~inside(backward, (sw, sh))] = 0
    save_rgb(out/'registered_rgb.png', registered)
    # Full geometric-overlap diagnostics show low-confidence warps too.
    # Additional reliable-only diagnostics are explicitly labelled.
    geometric = inside(backward, (sw, sh))
    overlay = target.copy()
    overlay[geometric] = np.rint((registered[geometric].astype(np.float32) + target[geometric])/2).astype(np.uint8)
    save_rgb(out/'overlay_50pct.png', overlay)
    reliable_overlay = target.copy()
    reliable_overlay[valid] = overlay[valid]
    save_rgb(out/'overlay_50pct_reliable.png', reliable_overlay)
    y, x = np.ogrid[:th, :tw]
    tiles = (x//metadata['checker_tile_px'] + y//metadata['checker_tile_px']) % 2 == 0
    choose = tiles & geometric
    checker = target.copy()
    checker[choose] = registered[choose]
    save_rgb(out/'checkerboard.png', checker)
    reliable_checker = target.copy()
    reliable_checker[tiles & valid] = registered[tiles & valid]
    save_rgb(out/'checkerboard_reliable.png', reliable_checker)
    Image.fromarray(valid.astype(np.uint8)*255).save(out/'valid_overlap_mask.png')
    Image.fromarray(inside(backward, (sw, sh)).astype(np.uint8)*255).save(out/'geometric_overlap_mask.png')
    np.save(out/'confidence_float32.npy', confidence.astype(np.float32))
    Image.fromarray(np.rint(np.clip(confidence,0,1)*65535).astype(np.uint16)).save(out/'confidence_16bit.png')
    Image.fromarray(np.rint(np.clip(confidence,0,1)*255).astype(np.uint8)).save(out/'confidence_preview.png')
    fields = {'target_to_source_xy_px': backward.astype(np.float32),
              'source_size_wh': np.array([sw,sh]), 'target_size_wh': np.array([tw,th]),
              'confidence_target': confidence.astype(np.float32), 'valid_target': valid}
    if forward is not None:
        fields['source_to_target_xy_px'] = forward.astype(np.float32)
        fields['confidence_source'] = fconfidence.astype(np.float32)
    if cycle is not None:
        fields['cycle_error_target_px'] = cycle.astype(np.float32)
    if determinant is not None:
        fields['jacobian_target_to_source'] = determinant.astype(np.float32)
    np.savez_compressed(out/'dense_transform.npz', **fields)
    # Flow visualization relative to the identity scaled for unequal dimensions.
    base = grid(th,tw)
    base[...,0] = (base[...,0]+.5)*sw/tw-.5
    base[...,1] = (base[...,1]+.5)*sh/th-.5
    displacement = backward-base
    mag, ang = cv2.cartToPolar(displacement[...,0], displacement[...,1])
    scale = max(float(np.percentile(mag,99)),1e-6)
    hsv = np.empty((th,tw,3),np.uint8)
    hsv[...,0] = np.mod(ang*90/np.pi,180).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = np.rint(np.clip(mag/scale,0,1)*255).astype(np.uint8)
    save_rgb(out/'dense_transform_preview.png',cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))
    metadata['flow_preview_magnitude_p99_source_px'] = scale
    metadata['effective_area_ratio'] = float(valid.mean())
    metadata['effective_pixel_count'] = int(valid.sum())
    metadata['geometric_area_ratio'] = float(inside(backward,(sw,sh)).mean())
    metadata['confidence_on_effective_area'] = stat(confidence, valid)
    gray1 = cv2.cvtColor(registered,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(target,cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Subsample only for diagnostic statistics, never for saved output images.
    sel = valid[::4,::4]
    metadata['gray_ncc'] = corr(gray1[::4,::4][sel],gray2[::4,::4][sel])
    grad1 = cv2.magnitude(cv2.Sobel(gray1,cv2.CV_32F,1,0),cv2.Sobel(gray1,cv2.CV_32F,0,1))
    grad2 = cv2.magnitude(cv2.Sobel(gray2,cv2.CV_32F,1,0),cv2.Sobel(gray2,cv2.CV_32F,0,1))
    metadata['gradient_ncc'] = corr(grad1[::4,::4][sel],grad2[::4,::4][sel])
    if cycle is not None:
        metadata['cycle_error_effective_target_px'] = stat(cycle,valid)
        metadata['cycle_error_confident_target_px'] = stat(cycle,inside(backward,(sw,sh)) & (confidence>=metadata['confidence_threshold']))
    metadata['quality_interpretation'] = 'Requires visual review; confidence/cycle consistency are internal checks, not independent ground truth. Seasonal RGB and root changes confound NCC.'
    (out/'metrics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def pair(model, srcpath, tgtpath, out, args):
    start=time.time()
    source,target=rgb(srcpath),rgb(tgtpath)
    sh,sw=source.shape[:2]; th,tw=target.shape[:2]
    label=f'{srcpath.stem}_to_{tgtpath.stem}'
    prealign=args.prealignments.get(label)
    if prealign is not None:
        prealign=np.asarray(prealign,np.float64)
        # This intermediate is used only for inference, never for final RGB.
        aligned=cv2.warpPerspective(source,prealign,(tw,th),flags=cv2.INTER_LINEAR)
        inference_source=Image.fromarray(aligned)
        inference_target=Image.fromarray(target)
        inferred_source_size=(tw,th)
    else:
        inference_source,inference_target=str(srcpath),str(tgtpath)
        inferred_source_size=(sw,sh)
    with torch.inference_mode():
        warp,certainty=model.match(inference_source,inference_target,device=args.device)
    # Current RoMa returns [B,H,2W,4], [B,H,2W] for symmetric mode.
    if warp.ndim==4:
        assert warp.shape[0]==1
        warp,certainty=warp[0],certainty[0]
    raw=warp.cpu().float().numpy(); rawc=certainty.cpu().float().numpy()
    mid=raw.shape[1]//2
    internal_forward=pixel_map(raw[:,:mid,2:],(tw,th),inferred_source_size)
    internal_backward=pixel_map(raw[:,mid:,:2],inferred_source_size,(tw,th))
    internal_cf=cv2.resize(rawc[:,:mid],inferred_source_size,interpolation=cv2.INTER_LINEAR)
    if prealign is not None:
        inverse=np.linalg.inv(prealign)
        pregrid=project(grid(sh,sw),prealign)
        forward=remap(internal_forward,pregrid,border=-1)
        backward=project(internal_backward,inverse)
        cf=remap(internal_cf,pregrid)
        cf[~inside(pregrid,(tw,th))]=0
    else:
        forward,backward,cf=internal_forward,internal_backward,internal_cf
    cb=cv2.resize(rawc[:,mid:],(tw,th),interpolation=cv2.INTER_LINEAR)
    assert np.isfinite(forward).all() and np.isfinite(backward).all()
    cyclic=remap(forward,backward,border=-1)
    cycle=np.linalg.norm(cyclic-grid(th,tw),axis=-1)
    f_at_b=remap(cf,backward)
    det=jacobian(backward)
    valid=(inside(backward,(sw,sh)) & (cb>=args.confidence) &
           (f_at_b>=args.confidence) & (cycle<=args.cycle) & (det>0))
    # Native source-grid cycle error used for sampled match filtering.
    fcycle=np.linalg.norm(remap(backward,forward,border=-1)-grid(sh,sw),axis=-1)
    fvalid=(inside(forward,(tw,th)) & (cf>=args.confidence) &
            (remap(cb,forward)>=args.confidence) &
            (fcycle <= args.cycle * max(sw/tw,sh/th)))
    torch.manual_seed(args.seed)
    matches,scores=model.sample(warp,certainty,num=args.samples)
    matches=matches.cpu().numpy(); scores=scores.cpu().numpy()
    xyA=(matches[:,:2]+1)*np.array([inferred_source_size[0]/2,inferred_source_size[1]/2])-.5
    if prealign is not None:
        xyA=project(xyA, np.linalg.inv(prealign))
    xyB=(matches[:,2:]+1)*np.array([tw/2,th/2])-.5
    smap=xyA.astype(np.float32)[None,:,:]
    tmap=xyB.astype(np.float32)[None,:,:]
    sc=remap(cf,smap)[0]; tc=remap(cb,tmap)[0]
    se=remap(fcycle,smap)[0]; te=remap(cycle,tmap)[0]
    ok=(inside(smap,(sw,sh))[0] & inside(tmap,(tw,th))[0] &
        (sc>=args.confidence) & (tc>=args.confidence) &
        (se<=args.cycle*max(sw/tw,sh/th)) & (te<=args.cycle))
    # Homography is diagnostic only; all output warps remain dense RoMa maps.
    hm=None; inliers=np.zeros(int(ok.sum()),bool)
    if ok.sum()>=4:
        hm,mask=cv2.findHomography(xyA[ok],xyB[ok],cv2.USAC_MAGSAC,args.ransac,maxIters=10000,confidence=.999)
        if mask is not None:inliers=mask.ravel().astype(bool)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'sampled_matches.npz',source_xy=xyA,target_xy=xyB,
                        roma_sample_score=scores,source_confidence=sc,target_confidence=tc,
                        reliable=ok,homography=hm if hm is not None else np.empty((0,0)),
                        homography_inliers_on_reliable=inliers)
    np.savez_compressed(out/'roma_raw_normalized.npz',warp=raw,confidence=rawc,
                        inference_source_size_wh=np.array(inferred_source_size),
                        prealignment_source_to_inference=prealign if prealign is not None else np.eye(3))
    meta={'pair':f'{srcpath.stem}_to_{tgtpath.stem}','kind':'adjacent',
          'source_file':str(srcpath),'target_file':str(tgtpath),
          'source_size_wh':[sw,sh],'target_size_wh':[tw,th],
          'model':'roma_outdoor','device':str(args.device),
          'coarse_resolution':args.coarse,'upsample_resolution':args.upsample,
          'confidence_threshold':args.confidence,'cycle_threshold_target_px':args.cycle,
          'checker_tile_px':args.tile,'sampled_matches':len(matches),
          'reliable_sampled_matches':int(ok.sum()),'reliable_dense_source_pixels':int(fvalid.sum()),
          'homography_inliers':int(inliers.sum()),
          'homography_inlier_ratio':float(inliers.mean()) if len(inliers) else None,
          'homography_ransac_target_px':args.ransac,
          'prealignment_source_to_inference':prealign.tolist() if prealign is not None else None,
          'foldover_target_ratio':float((det<=0).mean())}
    meta=output(out,source,target,backward,cb,valid,meta,forward,cf,cycle,det)
    meta['elapsed_seconds']=round(time.time()-start,2)
    (out/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    del source,target,warp,certainty,raw,rawc,forward,cf,cycle,fcycle,det
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    return backward,cb,valid,meta


def self_test():
    a=np.arange(5*7*3,dtype=np.uint8).reshape(5,7,3)
    g=grid(5,7)
    np.testing.assert_array_equal(remap(a,g),a)
    normalized=(g+.5)/np.array([7/2,5/2],np.float32)-1
    np.testing.assert_allclose(pixel_map(normalized,(7,5),(7,5)),g,atol=1e-6)
    np.testing.assert_allclose(jacobian(g),1)
    b=g.copy(); b[...,0]+=1
    np.testing.assert_array_equal(remap(a,b)[:,:-1],a[:,1:])
    # Composition: target->intermediate +1 x, intermediate->source +1 y.
    p=g.copy();p[...,1]+=1
    composed=remap(p,b,border=-1)
    np.testing.assert_allclose(composed[:-1,:-1],g[:-1,:-1]+[1,1])
    h=np.array([[1,0,2],[0,1,-1],[0,0,1]],np.float64)
    np.testing.assert_allclose(project(project(g,h),np.linalg.inv(h)),g,atol=1e-6)
    assert not inside(b,(7,5))[:,-1].any()
    print('PASS: identity, align_corners=False, translation direction, composition and out-of-bounds masks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--coarse',type=int,default=560)
    p.add_argument('--upsample',type=int,default=864)
    p.add_argument('--confidence',type=float,default=.5)
    p.add_argument('--cycle',type=float,default=3.,help='native target pixels')
    p.add_argument('--samples',type=int,default=5000)
    p.add_argument('--ransac',type=float,default=3.)
    p.add_argument('--tile',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--prealign-json',type=Path,help='Optional pair-name -> source-to-inference homography JSON')
    p.add_argument('--reuse-adjacent',action='store_true',help='Reuse matching existing adjacent outputs except prealigned pairs')
    args=p.parse_args()
    if args.self_test:self_test();return
    if args.input_dir is None or args.output_dir is None:p.error('Provide input and output directories')
    torch.set_num_threads(args.threads);cv2.setNumThreads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);cv2.setRNGSeed(args.seed)
    torch.set_float32_matmul_precision('highest')
    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.prealignments=json.loads(args.prealign_json.read_text()) if args.prealign_json else {}
    paths=[]
    for t in range(1,5):
        candidates=[f for f in args.input_dir.iterdir() if f.is_file() and f.stem.upper()==f'T{t}' and f.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff']]
        if len(candidates)!=1:raise ValueError(f'Expected exactly one T{t} RGB input: {candidates}')
        paths.append(candidates[0])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'input_sha256':{str(f):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in paths},
              'args':{k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(args).items()},
              'versions':{'torch':torch.__version__,'opencv':cv2.__version__},
              'coordinate_convention':'zero-based pixel centres, x then y; target_to_source_xy_px is the pullback used by remap; source_to_target_xy_px is the forward map',
              'interpolation':'bilinear for RGB and dense maps; nearest for binary masks; no colour correction or EXIF rotation',
              'overlap_definition':'in source bounds, bidirectional RoMa confidence >= threshold, cycle error <= target-pixel threshold, positive local Jacobian; confidence is not calibrated correctness probability',
              'cumulative_definition':'Tn->T1 pullback = composed adjacent backward maps; T1 RGB sampled once. Confidence is minimum along path, not a new RoMa inference. Validity intersects all adjacent masks.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    from romatch import roma_outdoor
    print(f'Loading full RoMa on {args.device}; inference {args.coarse}/{args.upsample}',flush=True)
    model=roma_outdoor(device=args.device,coarse_res=args.coarse,upsample_res=args.upsample,
                       symmetric=True,use_custom_corr=False,upsample_preds=True)
    cumulative=None;cum_conf=None;cum_mask=None;metrics=[]
    for i in range(3):
        name=f'T{i+1}_to_T{i+2}'
        folder=args.output_dir/name
        if args.reuse_adjacent and name not in args.prealignments and (folder/'metrics.json').exists():
            m=json.loads((folder/'metrics.json').read_text())
            assert m['confidence_threshold']==args.confidence and m['cycle_threshold_target_px']==args.cycle
            assert m['coarse_resolution']==args.coarse and m['upsample_resolution']==args.upsample
            with np.load(folder/'dense_transform.npz') as fields:
                b,c,v=fields['target_to_source_xy_px'],fields['confidence_target'],fields['valid_target']
            print(f'Reusing validated outputs: {name}',flush=True)
        else:
            b,c,v,m=pair(model,paths[i],paths[i+1],folder,args)
        metrics.append(m)
        if cumulative is None:cumulative,cum_conf,cum_mask=b,c,v
        else:
            previous_size=(cumulative.shape[1],cumulative.shape[0])
            new_map=remap(cumulative,b,border=-1)
            new_conf=np.minimum(c,remap(cum_conf,b))
            new_mask=v & inside(b,previous_size) & (remap(cum_mask.astype(np.uint8),b,nearest=True)>0)
            first=rgb(paths[0]);target=rgb(paths[i+1])
            new_mask &= inside(new_map,(first.shape[1],first.shape[0])) & (jacobian(new_map)>0) & (new_conf>=args.confidence)
            meta={'pair':f'T1_to_T{i+2}_cumulative','kind':'cumulative','source_size_wh':[first.shape[1],first.shape[0]],
                  'target_size_wh':[target.shape[1],target.shape[0]],'source_file':str(paths[0]),'target_file':str(paths[i+1]),
                  'checker_tile_px':args.tile,'confidence_threshold':args.confidence,
                  'sampled_matches':None,'reliable_sampled_matches':None,'homography_inliers':None,
                  'note':'Composed adjacent maps; no new direct matches and no independent match count.'}
            metrics.append(output(args.output_dir/f'T1_to_T{i+2}_cumulative',first,target,new_map,new_conf,new_mask,meta))
            cumulative,cum_conf,cum_mask=new_map,new_conf,new_mask
            del first,target;gc.collect()
        (args.output_dir/'summary.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['pair','kind','source_size_wh','target_size_wh','sampled_matches','reliable_sampled_matches','homography_inliers','effective_pixel_count','effective_area_ratio','geometric_area_ratio','gray_ncc','gradient_ncc']
    with (args.output_dir/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(metrics)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
