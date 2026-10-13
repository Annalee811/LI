#!/usr/bin/env python3
"""B2 seasonal REGION registration: retain new roots and local soil/root movement.
No local RoMa deformation is applied to RGB; confidence is never fabricated.
"""
from pathlib import Path
import os,json,csv,gc,argparse
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import cv2,numpy as np,h5py,torch
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u
from run_0353_1_root_priority import support
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'input/raw/V-变化/A017-1-B2';OLD=ROOT/'B2_shape_revision/final';OUT=ROOT/'B2_region_revision';F=OUT/'final';F.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(20261013);torch.set_num_threads(4);torch.manual_seed(20261013)
TS=[1,2,4];W=1600;ims={t:np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB')) for t in TS};wh={t:ims[t].shape[1::-1] for t in TS};oldpath=OLD/'evaluation.json';oldpath=oldpath if oldpath.exists() else ROOT.parent/'reference_prior/previous_evaluation.json';old=json.loads(oldpath.read_text());newrows=json.loads((OUT/'full_canvas_feature_search.json').read_text());candidate_path=ROOT/'B2_shape_revision/shape_feature_candidates.json';candidate_path=candidate_path if candidate_path.exists() else ROOT.parent/'reference_prior/previous_shape_feature_candidates.json';oldrows=json.loads(candidate_path.read_text());roots={t:support(ims[t])[0] for t in TS}
matrices={};summaries=[]
for a,b in [(1,2),(2,4),(1,4)]:
 name=f'T{a}_to_T{b}';prior=np.array(next(z['matrix_analysis_1600'] for z in old['feature_evaluation'] if z['pair']==name));candidates=[]
 for row in newrows:
  if row['pair']==name:candidates.extend(zip(row['source_points_analysis_1600'],row['target_points_analysis_1600']))
 for row in oldrows:
  if row['pair']==name:candidates.extend(zip(row['source_points'],row['target_points']))
 pa=[];pb=[];seen=set()
 for p,q in candidates:
  p,q=np.array(p),np.array(q);key=tuple(np.floor(p/4).astype(int))
  # Broad region hypothesis, not identical-root requirement: up to 120 T1 px local change is retained as candidate.
  if key in seen or np.linalg.norm(u.project(p,prior)-q)>75:continue
  seen.add(key);pa.append(p);pb.append(q)
 pa,pb=np.array(pa),np.array(pb);rt=[]
 for p,q in zip(pa,pb):
  p=np.rint(p*1400/1600).astype(int);q=np.rint(q*1400/1600).astype(int)
  rt.append(bool(roots[a][min(max(p[1],0),roots[a].shape[0]-1),min(max(p[0],0),1399)] and roots[b][min(max(q[1],0),roots[b].shape[0]-1),min(max(q[0],0),1399)]))
 rt=np.array(rt);cell=np.floor(pa/90).astype(int);_,inverse,counts=np.unique(cell,axis=0,return_inverse=True,return_counts=True);weight=np.where(rt,3.,1.)/np.sqrt(counts[inverse])
 # Fit stable structures robustly; changing descriptors remain in CSV and observational RGB.
 H,mask=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=50000,confidence=.999,refineIters=20);anchor=mask.ravel()>0;H=u.fit(pa[anchor],pb[anchor],weight[anchor]);error=np.linalg.norm(u.project(pa,H)-pb,axis=1)
 anchor=error<=10;H=u.fit(pa[anchor],pb[anchor],weight[anchor]);error=np.linalg.norm(u.project(pa,H)-pb,axis=1);scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]));assert .8<scale<1.2 and abs(angle)<10
 folds=(cell[:,0]+3*cell[:,1])%4;cv=np.full(len(pa),np.nan)
 for fold in range(4):
  train=(folds!=fold)&anchor;test=folds==fold
  if train.sum()>=6 and test.any():cv[test]=np.linalg.norm(u.project(pa[test],u.fit(pa[train],pb[train],weight[train]))-pb[test],axis=1)*wh[1][0]/W
 stable_cv=cv[anchor&np.isfinite(cv)];all_cv=cv[np.isfinite(cv)];span=np.ptp(pa[anchor],axis=0)*wh[1][0]/W
 summary=dict(pair=name,broad_region_candidates=len(pa),stable_structure_anchors=int(anchor.sum()),root_support_anchor_candidates=int((anchor&rt).sum()),spatial_cv_anchor_points=len(stable_cv),spatial_cv_anchor_median_T1_px=float(np.median(stable_cv)),spatial_cv_anchor_p95_T1_px=float(np.percentile(stable_cv,95)),spatial_cv_all_candidate_median_T1_px=float(np.median(all_cv)),stable_anchor_span_T1_xy=span.tolist(),normalized_uniform_scale=float(scale),rotation_deg=float(angle),matrix_analysis_1600=H.tolist(),manual_TRE=None,validation_interpretation='Automatic spatial-block CV conditional on descriptor-consensus anchor selection; all broader candidates retained separately; not human ground truth.')
 matrices[name]=H;summaries.append(summary)
 with (F/f'{name}_all_region_candidates.csv').open('w') as f:
  wr=csv.writer(f);wr.writerow(['source_x_analysis1600','source_y_analysis1600','target_x_analysis1600','target_y_analysis1600','root_support_proxy','stable_anchor','local_residual_T1_px','spatial_CV_T1_px'])
  for p,q,root,ok,e,cve in zip(pa,pb,rt,anchor,error*wh[1][0]/W,cv):wr.writerow([*p,*q,int(root),int(ok),float(e),float(cve)])
 print('REGION_ANCHORS',json.dumps(summary),flush=True)
# Consecutive available-period chain. The direct T1/T4 fit is corroboration, not replacement of the chain.
analysis={1:np.eye(3),2:matrices['T1_to_T2'],4:matrices['T2_to_T4']@matrices['T1_to_T2']};native={t:np.diag([wh[t][0]/W]*2+[1])@H@np.diag([W/wh[1][0]]*2+[1]) for t,H in analysis.items()}
h,w=ims[1].shape[:2];base=u.grid(h,w);maps={t:u.project(base,H) for t,H in native.items()};geom={t:u.inside(maps[t],wh[t]) for t in TS};aligned={t:u.remap(ims[t],maps[t]) for t in TS};common=np.logical_and.reduce(list(geom.values()));crop=u.rectangle(common);x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1];patch={t:aligned[t][sl] for t in TS}
for t in TS:
 Image.fromarray(aligned[t]).save(F/f'T{t}_full_T1_RGB.png',dpi=(600,600));Image.fromarray(patch[t]).save(F/f'T{t}_common_original_RGB.png',dpi=(600,600))
 # Actual source-pixel crop, without interpolation or colour changes.
 corners=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[t]);box=[max(0,int(np.floor(corners[:,0].min()))),max(0,int(np.floor(corners[:,1].min()))),min(wh[t][0],int(np.ceil(corners[:,0].max()))+1),min(wh[t][1],int(np.ceil(corners[:,1].max()))+1)];Image.fromarray(ims[t][box[1]:box[3],box[0]:box[2]]).save(F/f'T{t}_literal_native_original_crop.png',dpi=(600,600))
Image.fromarray(common.astype('uint8')*255).save(F/'common_observable_mask_T1.png');Image.fromarray(np.ones(patch[1].shape[:2],np.uint8)*255).save(F/'common_observable_mask_crop.png')
u.montage([patch[t] for t in TS],[f'T{t}：同一区域原色 RGB' for t in TS],F/'common_original_RGB_montage.png','保留新生、延伸与局部位移；T3缺失')
closure=np.linalg.norm(u.project(base[sl][::16,::16]*W/w,analysis[4])-u.project(base[sl][::16,::16]*W/w,matrices['T1_to_T4']),axis=-1)*w/W
D=dict(position='A017-1-B2',period_ids=TS,missing_period_ids=[3],T1_size_wh=[w,h],common_crop_xyxy=crop,common_crop_size_wh=[x1-x0,y1-y0],common_observable_pct_T1_full=100*common.mean(),common_crop_pct_T1_full=100*((x1-x0)*(y1-y0))/(w*h),matrices_T1_to_native={f'T{t}':H.tolist() for t,H in native.items()},region_features=summaries,chain_vs_direct_median_T1_px=float(np.median(closure)),chain_vs_direct_p95_T1_px=float(np.percentile(closure,95)),matching_goal='Same observational REGION across seasons; roots can be newly visible, extend, disappear, move; soil can change. Geometric visibility is not identical-content confidence.',RGB_rendering='Global similarity, once from source RGB; no elastic warping; no colour enhancement.',manual_landmark_TRE=None,previous_pixel_strict_success_ratio=old['automatic_pair_success_ratio'],previous_pixel_strict_common_pct_crop=old['common_effective_pct_crop'],previous_pixel_strict_interpretation='Old strict same-location point validation. Not a same-region registration failure rate.',pair_evaluation=[])
with h5py.File(F/'global_region_coordinate_fields.h5','w') as f:
 f.attrs['field_type']='Global similarity REGION alignment, original RGB only; not nonrigid RoMa';f.attrs['crop_T1_xyxy']=crop;f.attrs['period_ids']=TS
 for t in TS:
  g=f.create_group(f'T{t}');g.create_dataset('common_T1_grid_to_native_xy',data=maps[t][sl],compression='gzip',shuffle=True,fletcher32=True);g.create_dataset('T1_to_native_matrix',data=native[t]);g.attrs['native_size_wh']=wh[t]
(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2));print('REGION_CROP',crop,'closure',D['chain_vs_direct_median_T1_px'],flush=True)
# Observation windows carry biological appearance changes, not forced identity labels.
windows=[];font=ImageFont.truetype(u.FONT,27);small=ImageFont.truetype(u.FONT,20)
for win in old['observation_windows']:
 bx0,by0,bx1,by1=win['T1_xyxy'];bx0,bx1=np.clip([bx0,bx1],x0,x1);by0,by1=np.clip([by0,by1],y0,y1)
 if bx1>bx0 and by1>by0:windows.append(dict(label=win['label'],description=win['description'],T1_xyxy=[int(bx0),int(by0),int(bx1),int(by1)]))
heights=[];pimages=[]
for win in windows:
 bx0,by0,bx1,by1=win['T1_xyxy'];pis=[Image.fromarray(aligned[t][by0:by1,bx0:bx1]) for t in TS]
 for t,pi in zip(TS,pis):pi.save(F/f'T{t}_detail_{win["label"]}_RGB.png',dpi=(600,600))
 scale=min(1,650/pis[0].width,270/pis[0].height);size=(round(pis[0].width*scale),round(pis[0].height*scale));pimages.append([pi.resize(size,Image.Resampling.LANCZOS) for pi in pis]);heights.append(size[1]+55)
plate=Image.new('RGB',(2200,sum(heights)+95),'white');pd=ImageDraw.Draw(plate)
for k,t in enumerate(TS):pd.text((185+k*670,12),f'T{t}：同一区域的根系观察',font=font,fill='#22333c')
y=80
for win,pis,height in zip(windows,pimages,heights):
 pd.text((12,y+8),win['label'],font=font,fill='#245f75');pd.text((12,y+45),win['description'],font=small,fill='#43545e')
 for k,pi in enumerate(pis):plate.paste(pi,(185+k*670,y))
 y+=height
plate.save(F/'root_region_detail_comparison.png',dpi=(600,600));D['observation_windows']=windows
cli=argparse.ArgumentParser();cli.add_argument('--render-only',action='store_true');args=cli.parse_args()
if args.render_only:
 (F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2));raise SystemExit(0)
u.self_check()
from romatch import roma_outdoor
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=u.ContextRoMa(roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True));D['device']=str(device);D['RoMa_model']='roma_outdoor 560/864, overlapping 1000x900 context, fixed query ownership';common_supports=[];common_conf=[]
def infer_rgb(im):
 lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB);lab[...,0]=cv2.createCLAHE(2,(12,12)).apply(lab[...,0]);return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
for a,b in [(1,2),(2,4)]:
 name=f'T{a}_to_T{b}';out=F/name;out.mkdir(exist_ok=True);gmask=geom[a]&geom[b];maxrect=u.rectangle(gmask);box=[min(maxrect[0],x0),min(maxrect[1],y0),max(maxrect[2],x1),max(maxrect[3],y1)];bx0,by0,bx1,by1=box;psl=np.s_[by0:by1,bx0:bx1];A,B=aligned[a][psl],aligned[b][psl];ph,pw=A.shape[:2];pg=u.grid(ph,pw)
 for t,img in [(a,A),(b,B)]:Image.fromarray(img).save(out/f'T{t}_pair_original_RGB.png',dpi=(600,600))
 u.montage([A,B],[f'T{a} 原色 RGB',f'T{b} 原色 RGB'],out/'pair_original_RGB_montage.png','相同观察区域，允许根系与土壤发生变化')
 print('REGION_ROMA_START',name,box,(pw,ph),flush=True)
 forward,back,cf,cb=model.match(infer_rgb(A),infer_rgb(B),device=device)
 cycle=np.linalg.norm(u.remap(forward,back)-pg,axis=-1);bidir_conf=np.minimum(cb,u.remap(cf,back));jac=u.determinant(back);valid_points=gmask[psl]&u.remap(gmask[psl].astype('uint8'),back,nearest=True).astype(bool)&u.inside(back,(pw,ph))&(bidir_conf>=.5)&(cycle<=3)&(jac>0)
 displacement=back-pg;motion=np.linalg.norm(displacement,axis=-1);same_location=valid_points&(motion<=3);moved=valid_points&(motion>3)
 # Distinguish optical/geometric observation from actual neural point evidence.
 observation=gmask[psl];cb[~geom[b][psl]]=0;cf[~geom[a][psl]]=0
 for fname,mask in [('valid_observable_mask.png',observation),('confident_correspondence_mask.png',valid_points),('same_location_correspondence_mask.png',same_location),('local_displacement_correspondence_mask.png',moved)]:Image.fromarray(mask.astype('uint8')*255).save(out/fname)
 Image.fromarray(np.rint(cb*65535).astype('uint16')).save(out/'confidence_16bit.png');Image.fromarray(np.rint(cb*255).astype('uint8')).save(out/'confidence_preview.png')
 Image.fromarray(A).save(out/'registered_RGB.png',dpi=(600,600));overlay=np.rint((A.astype(float)+B)/2).astype('uint8');Image.fromarray(overlay).save(out/'overlay_50pct.png',dpi=(600,600));yy,xx=np.ogrid[:ph,:pw];choose=(yy//64+xx//64)%2==0;checker=B.copy();checker[choose]=A[choose];Image.fromarray(checker).save(out/'checkerboard.png',dpi=(600,600))
 pairH=native[b]@np.linalg.inv(native[a]);dispglobal=maps[b][psl]*w/wh[b][0]-maps[a][psl]*w/wh[a][0];flow=np.full((ph,pw,3),236,np.uint8)
 for y in range(50,ph,140):
  for x in range(50,pw,180):
   d=dispglobal[y,x];d=d/max(np.linalg.norm(d),1);cv2.arrowedLine(flow,(x,y),tuple(np.rint([x,y]+40*d).astype(int)),(51,94,119),2,tipLength=.2)
 Image.fromarray(flow).save(out/'global_transform_preview.png')
 local=B.copy()
 for y in range(28,ph,65):
  for x in range(28,pw,85):
   roi=valid_points[max(0,y-15):min(ph,y+16),max(0,x-15):min(pw,x+16)];dy,dx=np.where(roi)
   if len(dx)<30:continue
   ty=max(0,y-15)+dy;tx=max(0,x-15)+dx;delta=np.median(displacement[ty,tx],axis=0);mag=np.linalg.norm(delta)
   if mag<1:continue
   # Blue = target-query to source position; not a direction of root growth.
   end=np.array([x,y])+delta*min(1,35/max(mag,1));cv2.arrowedLine(local,(x,y),tuple(end.round().astype(int)),(185,222,233),1,tipLength=.18)
 Image.fromarray(local).save(out/'local_correspondence_displacement_preview.png',dpi=(600,600))
 with h5py.File(out/'RoMa_region_candidate_fields.h5','w') as f:
  f.attrs['field_type']='RoMa candidate point correspondence, allows local motion; NOT applied to final RGB';f.attrs['query_T1_crop_xyxy']=box;f.attrs['no_forced_same_location_threshold']=True;f.attrs['source_size_wh']=wh[a];f.attrs['target_size_wh']=wh[b];f.attrs['query_canvas_size_wh']=[pw,ph];f.attrs['RoMa_context_tiles']=json.dumps(model.tiles)
  data={'target_query_to_source_native_xy':u.project(back+[bx0,by0],native[a]),'source_query_to_target_native_xy':u.project(forward+[bx0,by0],native[b]),'confidence_target':cb,'confidence_source':cf,'cycle_error_T1_px':cycle,'local_displacement_target_to_source_T1_px':displacement,'confident_candidate_correspondence':valid_points,'same_location_correspondence_diagnostic':same_location,'geometric_observable_mask':observation}
  for key,value in data.items():f.create_dataset(key,data=value,compression='gzip',shuffle=True,fletcher32=True)
  f.create_dataset('global_source_to_target_native',data=pairH)
 # Deterministically thin dense evidence by spatial grid. Count is reproducible, not inflated by interpolation pixels.
 sites=[]
 for cy in range(0,ph,12):
  for cx in range(0,pw,12):
   mask=valid_points[cy:cy+12,cx:cx+12];ys,xs=np.where(mask)
   if not len(xs):continue
   scores=bidir_conf[cy+ys,cx+xs];j=int(np.argmax(scores));qy,qx=int(cy+ys[j]),int(cx+xs[j]);p=back[qy,qx];sites.append([qx,qy,float(p[0]),float(p[1]),float(cb[qy,qx]),float(cf[round(np.clip(p[1],0,ph-1)),round(np.clip(p[0],0,pw-1))]),float(motion[qy,qx]),bool(moved[qy,qx])])
 with (out/'spatially_thinned_RoMa_correspondences.csv').open('w') as f:
  wr=csv.writer(f);wr.writerow(['target_x_T1crop','target_y_T1crop','source_x_T1crop','source_y_T1crop','target_confidence','source_confidence','local_displacement_T1_px','locally_displaced_candidate']);wr.writerows(sites)
 fs=next(z for z in summaries if z['pair']==name);coverage=fs['stable_anchor_span_T1_xy'][0]/pw;vertical=fs['stable_anchor_span_T1_xy'][1]/ph
 region_supported=fs['stable_structure_anchors']>=20 and fs['spatial_cv_anchor_median_T1_px']<=3 and coverage>=.4 and vertical>=.2 and D['chain_vs_direct_median_T1_px']<=5
 met=dict(pair=name,calendar_gap_periods=b-a-1,pair_T1_crop_xyxy=box,largest_pair_geometric_rectangle_T1_xyxy=maxrect,pair_crop_size_wh=[pw,ph],pair_observable_pct_T1_full=100*gmask.mean(),pair_crop_pct_T1_full=100*(pw*ph)/(w*h),region_localization_supported=bool(region_supported),stable_anchor_horizontal_span_fraction=float(coverage),stable_anchor_vertical_span_fraction=float(vertical),spatially_thinned_confident_correspondences=len(sites),point_correspondence_support_pct_pair_crop=100*valid_points.mean(),same_location_point_support_pct_pair_crop=100*same_location.mean(),locally_displaced_point_support_pct_pair_crop=100*moved.mean(),low_confidence_pct_pair_crop=100*(cb<.5).mean(),confident_candidate_displacement_median_T1_px=float(np.median(motion[valid_points])) if valid_points.any() else None,interpretation='Region visibility retains new/extended roots and changed soil. Confident candidates with displacement >3 pixels are retained as possible local motion, not forced to coincide. Candidate motion is not proven soil migration/root growth; may include mismatch or global-pose estimation error.',point_support_rule='Bidirectional confidence >=0.5, cycle <=3 T1 px, positive Jacobian; no global-position-agreement <=3 constraint.',region_support_rule='>=20 stable auto anchors; spatial-CV anchor median<=3 T1 px; anchor x-span>=40% and y-span>=20% of pair crop; chain-vs-direct median<=5 T1 px; no minimum identical-content area.',manual_TRE=None)
 D['pair_evaluation'].append(met);(out/'evaluation.json').write_text(json.dumps(met,ensure_ascii=False,indent=2))
 # Crop common continuous canvas out of each pair; confidences are queried at the same original T1 coordinates.
 csl=np.s_[y0-by0:y1-by0,x0-bx0:x1-bx0];assert (0<=x0-bx0<x1-bx0<=pw and 0<=y0-by0<y1-by0<=ph);common_supports.append(valid_points[csl]);common_conf.append(cb[csl])
 panels=[(A,'配准原色 RGB：同一区域，保留季节变化'),(flow,'整体变换场预览：保持形状'),(np.repeat(np.rint(cb*255).astype('uint8')[...,None],3,2),'RoMa 置信度：低置信度≠不在同一区域'),(overlay,'50%叠加：允许局部错位与新生根'),(checker,'棋盘格：64个T1像素'),(np.repeat((observation*255).astype('uint8')[...,None],3,2),'有效共同视野：几何可观察；不是逐点匹配率')]
 six=Image.new('RGB',(pw,6*(ph+60)),'white');sd=ImageDraw.Draw(six)
 for i,(im,label) in enumerate(panels):sd.text((12,i*(ph+60)+10),name+'  '+label,font=font,fill='#26343d');six.paste(Image.fromarray(im),(0,i*(ph+60)+60))
 six.save(out/'six_results_composite.png',dpi=(600,600));print('REGION_ROMA_DONE',json.dumps(met),flush=True);del forward,back,cf,cb,cycle,displacement;gc.collect()
common_points=np.logical_and.reduce(common_supports);Image.fromarray((common_points*255).astype('uint8')).save(F/'common_confident_point_mask_crop.png');fullpoint=np.zeros((h,w),np.uint8);fullpoint[sl]=common_points*255;Image.fromarray(fullpoint).save(F/'common_confident_point_mask_T1.png')
D['common_point_correspondence_support_pct_crop']=100*common_points.mean();D['common_point_correspondence_support_pct_T1_full']=100*common_points.sum()/(w*h);D['region_localization_supported_pair_ratio']=sum(z['region_localization_supported'] for z in D['pair_evaluation'])/2
D['common_mask_semantics']='OBSERVABLE geometric region mask retains seasonal changes; POINT mask only represents RoMa correspondence evidence, not a root segmentation or same-region success rate.'
(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
with h5py.File(F/'global_region_coordinate_fields.h5','r') as f:
 for t in TS:
  H=f[f'T{t}/T1_to_native_matrix'][:];s=np.linalg.svd(H[:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10);np.testing.assert_array_equal(u.remap(ims[t],f[f'T{t}/common_T1_grid_to_native_xy'][:]),np.array(Image.open(F/f'T{t}_common_original_RGB.png')))
assert common[sl].all();print('PASS original RGB single sampling, no elastic deformation, shared geometric crop',flush=True)
