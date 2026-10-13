#!/usr/bin/env python3
"""B2 root-guided shape-preserving registration. RGB never uses a nonrigid field.
Run rematch_B2_shapes.py first, then this script, from any working directory.
"""
from pathlib import Path
import os,json,csv,gc,argparse
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
import cv2,numpy as np,h5py,torch
from PIL import Image,ImageDraw,ImageFont
import register_roma as r
import evaluate_continuous_registration as v
from run_0353_1_root_priority import support
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'input/raw/V-变化/A017-1-B2';OUT=ROOT/'B2_shape_revision';FINAL=OUT/'final';FINAL.mkdir(exist_ok=True)
cv2.setNumThreads(4);torch.set_num_threads(4);torch.manual_seed(20261013);cv2.setRNGSeed(20261013)
W=1600;TS=[1,2,4];ims={t:r.rgb(RAW/f'T{t}.jpg') for t in TS};wh={t:ims[t].shape[1::-1] for t in TS}
rows=json.loads((OUT/'shape_feature_candidates.json').read_text())

def fit(pa,pb,weights=None):
 weights=np.ones(len(pa)) if weights is None else weights
 X=np.zeros((2*len(pa),4));Y=pb.reshape(-1)
 X[::2,0]=pa[:,0];X[::2,1]=-pa[:,1];X[::2,2]=1
 X[1::2,0]=pa[:,1];X[1::2,1]=pa[:,0];X[1::2,3]=1
 sw=np.repeat(np.sqrt(weights),2);a,b,tx,ty=np.linalg.lstsq(X*sw[:,None],Y*sw,rcond=None)[0]
 return np.array([[a,-b,tx],[b,a,ty],[0,0,1]])

summaries=[];Hs={};features={}
for a,b in [(1,2),(2,4),(1,4)]:
 name=f'T{a}_to_T{b}';candidates=[d for d in rows if d['pair']==name and d.get('inliers',0)>=6 and .98<d.get('scale',0)<1.02 and abs(d.get('angle',99))<2]
 best=max(candidates,key=lambda d:d['inliers']);prior=np.array(best['matrix_analysis']);pa=[];pb=[];seen=set()
 for d in candidates:
  for p,q in zip(d['source_points'],d['target_points']):
   p,q=np.array(p),np.array(q)
   if np.linalg.norm(r.project(p,prior)-q)>5:continue
   key=tuple(np.floor(p/4).astype(int))
   if key in seen:continue
   seen.add(key);pa.append(p);pb.append(q)
 pa,pb=np.array(pa),np.array(pb)
 # Bright elongated support is only an analysis priority, never a root classification.
 ma,_=support(ims[a]);mb,_=support(ims[b]);root=[]
 for p,q in zip(pa,pb):
  pp=np.rint(p*1400/W).astype(int);qq=np.rint(q*1400/W).astype(int)
  root.append(bool(ma[min(pp[1],ma.shape[0]-1),min(pp[0],ma.shape[1]-1)] and mb[min(qq[1],mb.shape[0]-1),min(qq[0],mb.shape[1]-1)]))
 root=np.array(root);weights=np.where(root,3.,1.);H=fit(pa,pb,weights)
 err=np.linalg.norm(r.project(pa,H)-pb,axis=1);good=err<5;pa,pb,root,weights=pa[good],pb[good],root[good],weights[good];H=fit(pa,pb,weights)
 # Spatial block cross-validation: feature detections of a site do not straddle train/test.
 blocks=(np.floor(pa[:,0]/90).astype(int)+3*np.floor(pa[:,1]/90).astype(int))%4;cross=np.full(len(pa),np.nan)
 for fold in range(4):
  test=blocks==fold;train=~test
  if train.sum()>=6 and test.any():cross[test]=np.linalg.norm(r.project(pa[test],fit(pa[train],pb[train],weights[train]))-pb[test],axis=1)
 Hs[name]=H;features[name]=(pa,pb,root)
 finite=cross[np.isfinite(cross)]*wh[1][0]/W
 summary=dict(pair=name,unique_consensus_matches=len(pa),root_support_candidate_matches=int(root.sum()),fit_error_median_T1_px=float(np.median(np.linalg.norm(r.project(pa,H)-pb,axis=1))*wh[1][0]/W),spatial_cross_validation_points=len(finite),spatial_cross_validation_median_T1_px=float(np.median(finite)),spatial_cross_validation_p95_T1_px=float(np.percentile(finite,95)),normalized_uniform_scale=float(np.hypot(H[0,0],H[1,0])),rotation_degrees=float(np.degrees(np.arctan2(H[1,0],H[0,0]))),matrix_analysis_1600=H.tolist(),manual_landmark_TRE=None)
 summaries.append(summary)
 with (FINAL/f'{name}_feature_matches.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['source_x_analysis','source_y_analysis','target_x_analysis','target_y_analysis','root_support_proxy','spatial_cv_error_T1_px'])
  for p,q,rt,ce in zip(pa,pb,root,cross*wh[1][0]/W):writer.writerow([*p,*q,int(rt),float(ce)])
# Actual consecutive chain, independent T1/T4 direct fit is only closure check.
Hchain=Hs['T2_to_T4']@Hs['T1_to_T2'];Hdirect=Hs['T1_to_T4']
analysis={1:np.eye(3),2:Hs['T1_to_T2'],4:Hchain}
native={t:np.diag([wh[t][0]/W]*2+[1])@H@np.diag([W/wh[1][0]]*2+[1]) for t,H in analysis.items()}
h,w=ims[1].shape[:2];base=r.grid(h,w);maps={t:r.project(base,H) for t,H in native.items()};geoms={t:r.inside(maps[t],wh[t]) for t in TS};common=np.logical_and.reduce(list(geoms.values()));crop=v.rectangle(common);x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1]
aligned={t:r.remap(ims[t],maps[t]) for t in TS};patches={t:aligned[t][sl] for t in TS}
for t in TS:
 Image.fromarray(patches[t]).save(FINAL/f'T{t}_common_original_RGB.png',dpi=(600,600))
 Image.fromarray(aligned[t]).save(FINAL/f'T{t}_full_T1_canvas_RGB.png',dpi=(600,600))
Image.fromarray(common.astype('uint8')*255).save(FINAL/'common_geometric_mask_T1.png')
v.OUT=FINAL
v.montage([patches[t] for t in TS],[f'T{t} 原色 RGB' for t in TS],'cropped_RGB_montage.png','保持形状；T1坐标；T3缺失')
# User-guided observation regions, NOT supplied manual landmark coordinates.
windows=[('A',100,580,500,695,'上方长横根'),('B',160,690,640,785,'中部横根群'),('C',180,790,680,875,'下方横根群'),('D',640,590,1130,770,'右侧长根'),('E',460,830,800,990,'斜向分枝'),('F',690,850,910,1015,'陡斜根与邻根'),('G',1080,720,1420,905,'右侧细根'),('H',850,900,1400,1010,'下部边界与细根')]
window_meta=[];scene=Image.new('RGB',(3*(x1-x0),y1-y0+110),'white');sd=ImageDraw.Draw(scene);font=ImageFont.truetype(v.REG,26)
detail=Image.new('RGB',(1840,75+len(windows)*255),'white');dd=ImageDraw.Draw(detail)
for k,t in enumerate(TS):
 scene.paste(Image.fromarray(patches[t]),(k*(x1-x0),110));sd.text((k*(x1-x0)+15,12),f'T{t}  原色／统一尺度／保持根形状',font=font,fill='#24323d');dd.text((160+k*555,12),f'T{t}（T3缺失）',font=font,fill='#24323d')
for i,(label,ax0,ay0,ax1,ay1,text) in enumerate(windows):
 box=np.rint(np.array([ax0,ay0,ax1,ay1])*w/W).astype(int);box[[0,2]]=np.clip(box[[0,2]],x0,x1);box[[1,3]]=np.clip(box[[1,3]],y0,y1)
 bx0,by0,bx1,by1=box
 if bx1<=bx0 or by1<=by0:continue
 window_meta.append(dict(label=label,description=text,T1_xyxy=box.tolist(),kind='visual observation window; not manual correspondence ground truth'))
 dd.text((12,90+i*255),label,font=font,fill='#196080');dd.text((12,128+i*255),text[:5],font=ImageFont.truetype(v.REG,21),fill='#334455')
 for k,t in enumerate(TS):
  patch=aligned[t][by0:by1,bx0:bx1];Image.fromarray(patch).save(FINAL/f'T{t}_detail_{label}_RGB.png',dpi=(600,600))
  pim=Image.fromarray(patch);pim.thumbnail((545,230));detail.paste(pim,(160+k*555,65+i*255))
  sd.rectangle((k*(x1-x0)+bx0-x0,110+by0-y0,k*(x1-x0)+bx1-x0,110+by1-y0),outline='#5eabbf',width=2);sd.text((k*(x1-x0)+bx0-x0+4,110+by0-y0+2),label,font=font,fill='#ace8ff')
scene.save(FINAL/'root_structure_overview.png',dpi=(600,600));detail.save(FINAL/'root_detail_comparison.png',dpi=(600,600))
closure=np.linalg.norm(r.project(base[sl][::16,::16]*W/w,Hchain)-r.project(base[sl][::16,::16]*W/w,Hdirect),axis=-1)*w/W
summary=dict(position='A017-1-B2',period_ids=TS,missing_period_ids=[3],rendering='Global similarity only (uniform scaling, rotation, translation); RGB sampled once from originals; no elastic/dense RGB warping; no colour adjustment',T1_size_wh=[w,h],common_crop_xyxy=crop,common_crop_size_wh=[x1-x0,y1-y0],common_geometric_pct_T1=100*common.mean(),matrices_T1_to_native={f'T{t}':m.tolist() for t,m in native.items()},feature_evaluation=summaries,chain_vs_independent_direct_median_T1_px=float(np.median(closure)),chain_vs_independent_direct_p95_T1_px=float(np.percentile(closure,95)),observation_windows=window_meta,manual_landmark_TRE=None,limitations='Spatial CV checks automatic descriptor sites; root-support masks are proxies, not root identity or life/death labels. T2 has only 1330 native width; resampling cannot add captured detail. Direct and chain fit may share descriptors; closure is an internal check.')
(FINAL/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
with h5py.File(FINAL/'shape_preserving_coordinate_fields.h5','w') as f:
 f.attrs['field_type']='Global similarity RGB coordinate field; NOT RoMa nonrigid field';f.attrs['reference']='Original T1 zero-based (x,y) pixel centres';f.attrs['crop_xyxy']=crop;f.attrs['period_ids']=TS
 for t in TS:
  g=f.create_group(f'T{t}');g.create_dataset('T1_crop_to_native_xy_px',data=maps[t][sl],compression='gzip',shuffle=True,fletcher32=True);g.create_dataset('T1_to_native_matrix',data=native[t]);g.attrs['native_size_wh']=wh[t]
# Actual RoMa inference on the same T1 observation canvas; CLAHE is inference-only.
args=argparse.ArgumentParser();args.add_argument('--render-only',action='store_true');cli=args.parse_args()
if cli.render_only:raise SystemExit(0)
from romatch import roma_outdoor
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
from B2_tiled_roma import DetailedRoMa,coordinate_check
coordinate_check();model=DetailedRoMa(model)
summary['roma_inference']='Three overlapping 1000px root-detail context tiles, fixed disjoint query ownership, no confidence-max selection; CLAHE inference-only'
summary['device']=str(device);summary['roma_evaluation']=[];rh,rw=patches[1].shape[:2];g=r.grid(rh,rw);valids=[]
def inference(a):
 lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);lab[...,0]=cv2.createCLAHE(2,(12,12)).apply(lab[...,0]);return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
for a,b in [(1,2),(2,4)]:
 name=f'T{a}_to_T{b}';folder=FINAL/name;folder.mkdir(exist_ok=True);print('ROMA_START',name,device,patches[a].shape,flush=True)
 with torch.inference_mode():warp,conf=model.match(inference(patches[a]),inference(patches[b]),device=device)
 if warp.ndim==4:warp,conf=warp[0],conf[0]
 raw=warp.cpu().float().numpy();c=conf.cpu().float().numpy();mid=raw.shape[1]//2
 forward=r.pixel_map(raw[:,:mid,2:],(rw,rh),(rw,rh));backward=r.pixel_map(raw[:,mid:,:2],(rw,rh),(rw,rh));cf=cv2.resize(c[:,:mid],(rw,rh));cb=cv2.resize(c[:,mid:],(rw,rh));cycle=np.linalg.norm(r.remap(forward,backward,border=-1)-g,axis=-1);residual=np.linalg.norm(backward-g,axis=-1)
 # Effective for FINAL RGB requires RoMa bidirectional confidence, cycle, orientation AND agreement with global shape-preserving field.
 valid=(r.inside(backward,(rw,rh))&(cb>=.5)&(r.remap(cf,backward)>=.5)&(cycle<=3)&(r.jacobian(backward)>0)&(residual<=3));valids.append(valid)
 Image.fromarray(patches[a]).save(folder/'registered_RGB.png',dpi=(600,600));Image.fromarray(patches[b]).save(folder/'target_RGB.png',dpi=(600,600));Image.fromarray(np.rint((patches[a].astype(float)+patches[b])/2).astype('uint8')).save(folder/'overlay_50pct.png',dpi=(600,600))
 yy,xx=np.ogrid[:rh,:rw];choose=(yy//64+xx//64)%2==0;checker=patches[b].copy();checker[choose]=patches[a][choose];Image.fromarray(checker).save(folder/'checkerboard.png',dpi=(600,600))
 Image.fromarray(valid.astype('uint8')*255).save(folder/'effective_mask.png');Image.fromarray(np.rint(cb*65535).astype('uint16')).save(folder/'confidence_16bit.png');Image.fromarray(np.rint(cb*255).astype('uint8')).save(folder/'confidence_preview.png')
 # Global display displacement in normalized T1-size pixel units, not arbitrary rainbow nonrigid deformation.
 pairH=native[b]@np.linalg.inv(native[a]);srcmap=maps[a][sl];trgmap=maps[b][sl];disp=(trgmap*wh[1][0]/wh[b][0])-(srcmap*wh[1][0]/wh[a][0]);mag,ang=cv2.cartToPolar(disp[...,0],disp[...,1]);hsv=np.zeros((rh,rw,3),'uint8');hsv[...,0]=(ang*90/np.pi).astype('uint8')%180;hsv[...,1]=120;hsv[...,2]=180;flow=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
 for y in range(48,rh,120):
  for x in range(48,rw,180):
   direction=disp[y,x]/max(mag[y,x],1);cv2.arrowedLine(flow,(x,y),tuple(np.rint([x,y]+35*direction).astype(int)),(30,30,30),1,tipLength=.25)
 Image.fromarray(flow).save(folder/'transform_field_preview.png')
 # Native coordinate RoMa candidate maps: sampling coordinates only, NEVER used to render RGB.
 with h5py.File(folder/'RoMa_candidate_fields.h5','w') as f:
  f.attrs['field_type']='RoMa nonrigid candidate correspondence diagnostic ONLY; final RGB uses global similarity';f.attrs['reference']='target query lies on the shared T1 crop canvas';f.attrs['T1_crop_xyxy']=crop
  data={'target_query_to_source_native_xy_px':r.project(backward+[x0,y0],native[a]),'source_query_to_target_native_xy_px':r.project(forward+[x0,y0],native[b]),'confidence_target_query':cb,'confidence_source_query':cf,'cycle_error_T1_px':cycle,'global_agreement_error_T1_px':residual,'effective_for_global_RGB':valid,'raw_normalized_warp':raw,'raw_confidence':c}
  for key,value in data.items():f.create_dataset(key,data=value,compression='gzip',shuffle=True,fletcher32=True)
  f.create_dataset('global_source_native_to_target_native',data=pairH)
 with torch.inference_mode():matches,scores=model.sample(warp,conf,num=10000)
 matches=matches.cpu().numpy();p=(matches[:,:2]+1)*[rw/2,rh/2]-.5;q=(matches[:,2:]+1)*[rw/2,rh/2]-.5
 confidence_s=r.remap(cf,p.astype('float32')[None])[0];confidence_t=r.remap(cb,q.astype('float32')[None])[0];point_error=np.linalg.norm(p-q,axis=-1);ok=(confidence_s>=.5)&(confidence_t>=.5)&(point_error<=3)&(r.remap(valid.astype('uint8'),q.astype('float32')[None],nearest=True)[0]>0)
 with (folder/'RoMa_candidate_matches.csv').open('w') as f:
  wr=csv.writer(f);wr.writerow(['source_x_T1crop','source_y_T1crop','target_x_T1crop','target_y_T1crop','confidence_source','confidence_target','global_agreement_T1_px','effective'])
  for pp,qq,cs,ct,err,k in zip(p,q,confidence_s,confidence_t,point_error,ok):wr.writerow([*pp,*qq,float(cs),float(ct),float(err),int(k)])
 fs=next(z for z in summaries if z['pair']==name);accepted=len(p[ok])>=100 and valid.mean()>=.1 and fs['spatial_cross_validation_median_T1_px']<=3
 met=dict(pair=name,calendar_gap_periods=b-a-1,sampled_matches=len(p),effective_sampled_matches=int(ok.sum()),effective_pct_common_canvas=100*valid.mean(),effective_pct_T1_full=100*valid.sum()/(w*h),low_confidence_pct_common_canvas=100*(cb<.5).mean(),global_agreement_error_confident_T1_px=r.stat(residual,cb>=.5),effective_cycle_error_T1_px=r.stat(cycle,valid),automatic_global_registration_pass=bool(accepted),effective_definition='Bidirectional RoMa confidence >=0.5; cycle <=3 T1 pixels; positive candidate Jacobian; candidate displacement from GLOBAL RGB <=3 T1 pixels',manual_landmark_TRE=None)
 summary['roma_evaluation'].append(met);(folder/'evaluation.json').write_text(json.dumps(met,indent=2))
 # Six-row common-crop figures preserve native crop pixel scale.
 canvas=Image.new('RGB',(rw,rh*6+6*60),'white');d=ImageDraw.Draw(canvas)
 panels=[(patches[a],'配准 RGB：整体等比缩放、旋转、平移'),(flow,'全局变换场预览（用于 RGB）'),(np.repeat(np.rint(cb*255).astype('uint8')[...,None],3,2),'RoMa 置信度（0–1）'),(np.rint((patches[a].astype(float)+patches[b])/2).astype('uint8'),'50% 透明叠加'),(checker,'棋盘格（64 T1 像素）'),(np.repeat((valid*255).astype('uint8')[...,None],3,2),'有效掩膜：RoMa 可信且与全局位置相符')]
 for i,(im,label) in enumerate(panels):d.text((15,i*(rh+60)+10),name+'  '+label,font=font,fill='#26333e');canvas.paste(Image.fromarray(im),(0,i*(rh+60)+60))
 canvas.save(folder/'six_results_composite.png',dpi=(600,600));print('ROMA_DONE',json.dumps(met),flush=True);del warp,conf,raw,c,forward,backward;gc.collect()
commonvalid=np.logical_and.reduce(valids);Image.fromarray((commonvalid*255).astype('uint8')).save(FINAL/'common_effective_mask_crop.png')
fullvalid=np.zeros((h,w),np.uint8);fullvalid[sl]=commonvalid*255;Image.fromarray(fullvalid).save(FINAL/'common_effective_mask_T1.png')
summary['common_effective_pct_T1']=100*commonvalid.sum()/(w*h);summary['common_effective_pct_crop']=100*commonvalid.mean();summary['automatic_pair_success_ratio']=sum(z['automatic_global_registration_pass'] for z in summary['roma_evaluation'])/2
summary['quality_rules']='Automatic pass: >=100 effective sampled matches; >=10% of geometric common crop effective; spatial-CV median <=3 T1 pixels. No manual TRE. No inferred root death.'
(FINAL/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with h5py.File(FINAL/'shape_preserving_coordinate_fields.h5','r') as f:
 for t in TS:np.testing.assert_array_equal(r.remap(ims[t],f[f'T{t}/T1_crop_to_native_xy_px'][:]),np.array(Image.open(FINAL/f'T{t}_common_original_RGB.png')))
assert common[sl].all();print('PASS single-sample original RGB / largest shared rectangle / uniform similarity',flush=True)
