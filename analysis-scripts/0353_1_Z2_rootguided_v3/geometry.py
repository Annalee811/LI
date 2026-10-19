"""Root-salient RoMa similarities inside user-guided bands; dense RGB warps forbidden."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from skimage.filters import frangi
SUPPORT=Path(__file__).parent/'support'
if not SUPPORT.exists():SUPPORT=Path(__file__).parent.parent/'all_series_localization_v2'
sys.path.insert(0,str(SUPPORT))
from localize_q1 import modes,project
from assemble_q1 import largest_rect
from render_results import sample,grid

def roots(im):
 im=im.copy();im.thumbnail((1600,2000));g=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY)/255.;v=frangi(g,sigmas=[1,2,3],black_ridges=False);return np.clip(v/max(np.quantile(v,.99),1e-8),0,1).astype('float32')

def native_matrix(H,wa,wb):return np.diag([wb/1600,wb/1600,1])@H@np.diag([1600/wa,1600/wa,1])

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();work=a.work;pos='0353-1-Z2';cv2.setNumThreads(1);cv2.setRNGSeed(20261019)
 pri=json.loads((work/'user_region_priors.json').read_text());wh={t:pri[f'T{t}']['original_wh'] for t in range(1,5)};boxes={t:pri[f'T{t}']['native_box'] for t in range(1,5)};rootmaps={};crops={}
 for t in range(1,5):
  im=Image.open(work/f'search/inputs/raw/{pos}/T{t}.png').convert('RGB');crops[t]=im.size;rootmaps[t]=roots(im)
 desc=json.loads((work/'descriptor_evidence/descriptor_search.json').read_text());descriptor_models={};descriptor_points={}
 for key,models in desc['combined_hypotheses'].items():
  aa,bb=[int(s[1:]) for s in key.split('_to_')];Ca=np.array([[crops[aa][0]/1600,0,boxes[aa][0]],[0,crops[aa][0]/1600,boxes[aa][1]],[0,0,1]]);Cb=np.array([[crops[bb][0]/1600,0,boxes[bb][0]],[0,crops[bb][0]/1600,boxes[bb][1]],[0,0,1]])
  descriptor_models[key]=[dict(m,native_matrix=(Cb@np.array(m['matrix_analysis1600'])@np.linalg.inv(Ca)).tolist()) for m in models]
  with np.load(work/'descriptor_evidence'/(key+'_descriptor_matches.npz')) as z:pa=z['source_xy_analysis1600']*crops[aa][0]/1600+boxes[aa][:2];pb=z['target_xy_analysis1600']*crops[bb][0]/1600+boxes[bb][:2]
  descriptor_points[key]=(pa,pb)
 records=[];selected={}
 for aa,bb in [(1,2),(2,3),(3,4)]:
  key=f'T{aa}_to_T{bb}';ep=work/'search/roma_evidence'/pos/key
  with np.load(ep/'dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
  with np.load(ep/'sampled_matches.npz') as z:pa=z['source_xy'];pb=z['target_xy'];ca=z['source_confidence']
  cb=sample(d['target_confidence'],pb.reshape(1,-1,2),crops[bb]).ravel();back=(sample(d['backward'],pb.reshape(1,-1,2),crops[bb])[0]+1)*[crops[aa][0]/2,crops[aa][1]/2]-.5;cycle=np.linalg.norm(back-pa,axis=1)*1600/wh[aa][0];cc=np.minimum(ca,cb)
  sa=sample(rootmaps[aa],pa.reshape(1,-1,2),crops[aa]).ravel();sb=sample(rootmaps[bb],pb.reshape(1,-1,2),crops[bb]).ravel();pna=pa+boxes[aa][:2];pnb=pb+boxes[bb][:2];qa=pna*1600/wh[aa][0];qb=pnb*1600/wh[bb][0]
  good=(cc>=.3)&(cycle<=10);root=good&(sa>=.1)&(sb>=.1);root_modes=modes(qa[root],qb[root],minimum=12);all_modes=modes(qa[good],qb[good],minimum=12)
  candidates=[]
  for source,ms in [('root_RoMa',root_modes),('all_RoMa',all_modes)]:
   for m in ms:
    H=native_matrix(np.array(m['matrix_analysis1600']),wh[aa][0],wh[bb][0]);candidates.append(dict(m,source=source,native_matrix=H.tolist()))
  candidates.extend(dict(m,source='RootSIFT') for m in descriptor_models[key])
  # Prespecified order: root-salient neural evidence, full neural evidence, descriptors.
  eligible=[m for m in candidates if m['anchors']>=20 and m['source_span_xy'][0]>=400 and .8<np.linalg.norm(np.array(m['native_matrix'])[:2,0])*wh[aa][0]/wh[bb][0]<1.2]
  chosen=eligible[0] if eligible else None
  checked=None
  if chosen is not None:
   H=np.array(chosen['native_matrix']);sel=descriptor_models[key];dp,dq=descriptor_points[key]
   if sel:
    DH=np.array(sel[0]['native_matrix']);ok=np.linalg.norm(project(dp,DH)-dq,axis=1)<=5*crops[bb][0]/1600;errors=np.linalg.norm(project(dp[ok],H)-dq[ok],axis=1)
    checked=dict(n=int(ok.sum()),median_error_native_target_px=float(np.median(errors)) if len(errors) else None,RMSE_native_target_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,selection='Held from neural fitting, conditional on earlier descriptor consensus; not manual TRE.')
   selected[key]=chosen
  rec=dict(pair=key,total_sampled_points=len(pa),bidirectional_c_ge_0_3_cycle_le_10_analysis1600=int(good.sum()),root_salient_eligible_points=int(root.sum()),max_sampled_bidirectional_confidence=float(cc.max()),candidates=candidates,selected=chosen,automatic_descriptor_check=checked,manual_landmark_TRE=None);records.append(rec)
  np.savez_compressed(work/(key+'_root_points.npz'),source_native=pna[root],target_native=pnb[root],confidence=cc[root]);print('GEOMETRY',key,'good',good.sum(),'roots',root.sum(),'selected',None if chosen is None else(chosen['source'],chosen['anchors'],chosen['source_span_xy']), 'check',checked,flush=True)
 audit=dict(position=pos,priors=pri,pair_records=records,model_rule='Prespecified root-ridge saliency>=0.1 on both ends, bidirectional confidence>=0.3, cycle<=10 at analysis width1600; then coherent uniform similarity. Prefer eligible root-RoMa, then all-RoMa, then descriptor evidence.',manual_TRE=None)
 (work/'geometry_audit.json').write_text(json.dumps(audit,indent=2))
 if 'T2_to_T3' not in selected and (work/'coarse_bridge.json').exists():
  coarse=json.loads((work/'coarse_bridge.json').read_text());peaks=coarse['peaks']
  if peaks[0]['ridge_NCC']>=.1 and peaks[0]['ridge_NCC']-peaks[1]['ridge_NCC']>=.05:
   fallback=dict(source='user_guided_root_ridge_coarse_translation',anchors=0,native_matrix=coarse['native_matrix'],status='COARSE_UNVERIFIED',mandatory_review=True,coarse_evidence=coarse)
   selected['T2_to_T3']=fallback
   next(v for v in records if v['pair']=='T2_to_T3')['selected']=fallback
   print('COARSE_FALLBACK_NOT_POINT_SUCCESS',coarse['native_matrix'],flush=True)
 if len(selected)!=3:raise RuntimeError('At least one adjacent link lacks enough reliable evidence. Do not synthesize a connected sequence.')
 poses={1:np.eye(3)}
 for aa,bb in [(1,2),(2,3),(3,4)]:poses[bb]=np.array(selected[f'T{aa}_to_T{bb}']['native_matrix'])@poses[aa]
 q=grid([0,0,*wh[1]]);b=boxes[1];valid=(q[...,0]>=b[0])&(q[...,0]<=b[2]-1)&(q[...,1]>=b[1])&(q[...,1]<=b[3]-1)
 for t in [2,3,4]:
  qq=project(q,poses[t]);b=boxes[t];valid&=(qq[...,0]>=b[0])&(qq[...,0]<=b[2]-1)&(qq[...,1]>=b[1])&(qq[...,1]<=b[3]-1)
 box=largest_rect(valid);assert (box[2]-box[0])>=wh[1][0]*.5 and box[3]-box[1]>=80,box
 # Direct descriptor routes were never used to compose these three adjacent similarities.
 checks=[];corn=np.array([[box[0],box[1]],[box[2]-1,box[1]],[box[0],box[3]-1],[box[2]-1,box[3]-1],[(box[0]+box[2])/2,(box[1]+box[3])/2]])
 for aa,bb in [(1,3),(1,4),(2,4)]:
  key=f'T{aa}_to_T{bb}';ms=descriptor_models[key]
  if not ms:checks.append(dict(pair=key,available=False));continue
  pa=project(corn,poses[aa]);pb=project(corn,poses[bb]);errors=[float(np.median(np.linalg.norm(project(pa,np.array(m['native_matrix']))-pb,axis=1)))*1600/wh[bb][0] for m in ms];ix=int(np.argmin(errors));checks.append(dict(pair=key,available=True,minimum_route_disagreement_analysis1600=errors[ix],descriptor_anchors=ms[ix]['anchors'],conditional_check=True))
 review=any(c.get('available') and c['minimum_route_disagreement_analysis1600']>10 for c in checks) or any(v['selected']['source']!='root_RoMa' for v in records)
 region=dict(id='A',box=box,matrices={f'T{t}':H.tolist() for t,H in poses.items()},review_required=review,source='Fresh user-guided root-band RoMa; adjacent uniform similarities; original colors; no dense/nonrigid RGB warps',evidence=dict(adjacent_pair_records=records,redundant_descriptor_route_checks=checks,manual_TRE=None))
 entry=dict(position=pos,periods=[1,2,3,4],regions=[region],scope='Only 0353-1-Z2; root band indicated by user; T1 common coordinates',input_sha256={k:v['original_sha256'] for k,v in pri.items()})
 audit['common_T1_box']=box;audit['poses_T1_to_native']={f'T{t}':H.tolist() for t,H in poses.items()};audit['redundant_route_checks']=checks;audit['review_required']=review;(work/'geometry_audit.json').write_text(json.dumps(audit,indent=2));(work/'registry.json').write_text(json.dumps({pos:entry},indent=2));print('COMMON_REGION',box,'review',review,'routes',checks,flush=True)
if __name__=='__main__':main()
