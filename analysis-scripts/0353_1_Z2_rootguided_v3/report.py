"""Export original-color common RGB, six QC types, dense fields and honest quality tables."""
import argparse,csv,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/LI_Z2_plot_cache')
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SUPPORT=Path(__file__).parent/'support'
if not SUPPORT.exists():SUPPORT=Path(__file__).parent.parent/'all_series_localization_v2'
sys.path.insert(0,str(SUPPORT));from render_results import sample,proj,grid,csvwrite

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=a.work;pos='0353-1-Z2';out=r/'deliverable'/pos;cv2.setNumThreads(1);m=json.loads((out/'geometry_metadata.json').read_text());audit=json.loads((r/'geometry_audit.json').read_text());pri=json.loads((r/'user_region_priors.json').read_text());box=m['regions'][0]['box'];H={int(k[1:]):np.array(v) for k,v in m['regions'][0]['matrices'].items()};images={t:np.array(Image.open(out/f'RGB/T{t}_independent_stitched_RGB.png').convert('RGB')) for t in [1,2,3,4]};hh,ww=images[1].shape[:2];q=grid([0,0,ww,hh]);yy,xx=np.mgrid[:hh,:ww]
 for name in ['QC','fields','native_crops','evaluation']:(out/name).mkdir(exist_ok=True)
 # Native, unresampled bbox crops complement the four common-scale images.
 for t in [1,2,3,4]:
  raw=np.array(Image.open(next((out/'originals').glob(f'T{t}.*'))).convert('RGB'));poly=np.array(m['polygons_native_pixel_centres'][f'T{t}']['A']);lo=np.maximum(np.floor(poly.min(0)).astype(int),0);hi=np.minimum(np.ceil(poly.max(0)).astype(int)+1,raw.shape[1::-1]);x0,y0=lo;x1,y1=hi;Image.fromarray(raw[y0:y1,x0:x1]).save(out/f'native_crops/T{t}_unresampled_native_RGB_bbox.png',dpi=(600,600));mask=np.array(Image.open(out/f'masks/T{t}_native_geometric_overlap_mask.png'));Image.fromarray(mask[y0:y1,x0:x1]).save(out/f'native_crops/T{t}_native_bbox_common_polygon_mask.png')
 rows=[];panels=[]
 for aa,bb in [(1,2),(2,3),(3,4)]:
  key=f'T{aa}_to_T{bb}';dest=out/'QC'/key;dest.mkdir(exist_ok=True);A=images[aa];B=images[bb];ep=r/'posed_roma_evidence'/pos/key
  with np.load(ep/'dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
  pred=(sample(d['forward'],q,d['source_wh'])+1)*d['target_wh']/2-.5;cf=np.minimum(sample(d['source_confidence'],q,d['source_wh']),sample(d['target_confidence'],pred,d['target_wh'])).clip(0,1);back=(sample(d['backward'],pred,d['target_wh'])+1)*d['source_wh']/2-.5;cycle=np.linalg.norm(back-q,axis=-1);flow=pred-q;motion=np.linalg.norm(flow,axis=-1);owner=(pred[...,0]>=0)&(pred[...,0]<=ww-1)&(pred[...,1]>=0)&(pred[...,1]<=hh-1);cf*=owner;eff=owner&(cf>=.5)&(cycle<=10)&(motion<=15);overlay=np.rint((A.astype(float)+B)/2).astype('uint8');checker=np.where((((xx//64+yy//64)%2)==0)[...,None],A,B);cimg=np.uint8(np.rint(plt.get_cmap('cividis')(cf)[...,:3]*255))
  for name,im in [('registered_RGB_current_in_common_T1',B),('previous_RGB_in_common_T1',A),('confidence_preview',cimg),('overlay_50pct',overlay),('checkerboard',checker),('effective_correspondence_mask',np.uint8(eff)*255),('geometric_observation_mask',np.full((hh,ww),255,np.uint8))]:Image.fromarray(im).save(dest/(name+'.png'),dpi=(600,600))
  native_source=proj(q+box[:2],H[aa]);native_target=proj(pred+box[:2],H[bb]);G=H[bb]@np.linalg.inv(H[aa]);np.savez_compressed(out/'fields'/(key+'_dense_fields_float32.npz'),query_common_T1_xy=q,target_common_T1_xy=pred.astype('float32'),displacement_common_T1_xy=flow.astype('float32'),source_native_xy=native_source,target_native_xy=native_target,confidence=cf.astype('float32'),cycle_error_common_T1_px=cycle.astype('float32'),global_similarity_previous_native_to_current_native=G,T1_common_box_xyxy=np.array(box))
  Image.fromarray(np.uint16(np.rint(cf*65535))).save(dest/'confidence_16bit.png')
  fig,ax=plt.subplots(figsize=(7,2.8),layout='constrained');ax.imshow(A);step=max(40,ww//35);sl=(slice(None,None,step),slice(None,None,step));keep=(cf[sl]>=.3)&(motion[sl]<=80)
  ax.quiver(xx[sl][keep],yy[sl][keep],flow[sl][...,0][keep],flow[sl][...,1][keep],cf[sl][keep],cmap='cividis',clim=(0,1),angles='xy',scale_units='xy',scale=1,width=.003)
  if not keep.any():ax.text(.02,.04,'No vectors meet display threshold',transform=ax.transAxes,color='white',bbox=dict(facecolor='black',alpha=.5))
  ax.set_title(f'T{aa} → T{bb} | residual dense flow (T1 px)',fontsize=10);ax.axis('off');fig.savefig(dest/'transform_field_preview.png',dpi=250);plt.close(fig)
  panels.append([B,np.array(Image.open(dest/'transform_field_preview.png')),cimg,overlay,checker,np.uint8(eff)*255]);rec=next(v for v in audit['pair_records'] if v['pair']==key);sel=rec['selected'];check=rec['automatic_descriptor_check'];rows.append(dict(pair=key,localization_method=sel['source'],point_based_automatic_localization_supported=sel['source']!='user_guided_root_ridge_coarse_translation',coarse_user_region_guidance_required=sel['source']=='user_guided_root_ridge_coarse_translation',original_view_bidirectional_sampled_matches=rec['bidirectional_c_ge_0_3_cycle_le_10_analysis1600'],root_salient_sampled_matches=rec['root_salient_eligible_points'],similarity_fit_anchors=sel['anchors'],conditional_neural_spatial_CV_median_analysis1600_px=sel.get('conditional_spatial_CV_median_analysis1600'),descriptor_check_n=check['n'] if check else 0,conditional_descriptor_median_native_target_px=check['median_error_native_target_px'] if check else None,manual_landmark_TRE=None,geometric_observation_pixels=hh*ww,pose_conditioned_high_confidence_pixels=int(eff.sum()),pose_conditioned_high_confidence_percent=100*float(eff.mean()),pose_conditioned_low_confidence_percent=100*float((cf<.5).mean()),pose_conditioned_mean_confidence=float(cf.mean()),QC_confidence_threshold=.5,QC_cycle_limit_T1_px=10,QC_displacement_limit_T1_px=15))
 # Six categories × three adjacent transitions.
 fig,axs=plt.subplots(3,6,figsize=(21,8.0),layout='constrained');titles=['Aligned current RGB','Dense transform preview','RoMa confidence','50% overlay','Checkerboard','Correspondence mask']
 for row,ims in enumerate(panels):
  for col,im in enumerate(ims):
   ax=axs[row,col];ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title((f'T{row+1} → T{row+2}\n' if col==0 else '')+titles[col],fontsize=9)
   if col==2:fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='cividis'),ax=ax,fraction=.025,pad=.02)
 fig.suptitle('0353-1-Z2 | user-guided root observation region | T2→T3 coarse localization requires review',fontsize=14);fig.savefig(out/'QC/0353-1-Z2_six_category_overview.png',dpi=300);fig.savefig(out/'QC/0353-1-Z2_six_category_overview.pdf');plt.close(fig)
 # Four-period RGB and an identical local window at native T1 pixel scale.
 fig,axs=plt.subplots(2,4,figsize=(16,6.4),layout='constrained');local=[int(ww*.50),int(hh*.23),int(ww*.82),int(hh*.88)]
 for col,t in enumerate([1,2,3,4]):
  im=images[t];axs[0,col].imshow(im);axs[0,col].set_title(f'T{t} | full common window');x0,y0,x1,y1=local;axs[1,col].imshow(im[y0:y1,x0:x1]);axs[1,col].set_title(f'T{t} | identical local window')
 for ax in axs.flat:ax.axis('off')
 fig.suptitle('0353-1-Z2 | original colors | T1 pixel scale | T2→T3 provisional',fontsize=14);fig.savefig(out/'QC/four_period_RGB_full_and_local.png',dpi=300);fig.savefig(out/'QC/four_period_RGB_full_and_local.pdf');plt.close(fig)
 csvwrite(out/'evaluation/adjacent_registration_evaluation.csv',rows)
 import shutil
 for name in ['geometry_audit.json','user_region_priors.json','coarse_bridge.json']:
  shutil.copy2(r/name,out/'evaluation'/name)
 attempts=[]
 for folder in [r/'search/roma_evidence'/pos,r/'tight_gray/roma_evidence'/pos]:
  for ep in sorted(folder.glob('T*_to_T*')):
   if not(ep/'completed.json').exists():continue
   with np.load(ep/'dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
   with np.load(ep/'sampled_matches.npz') as z:pa=z['source_xy'];pb=z['target_xy'];ca=z['source_confidence']
   aa=int(ep.name[1]);cb=sample(d['target_confidence'],pb.reshape(1,-1,2),d['target_wh']).ravel();back=(sample(d['backward'],pb.reshape(1,-1,2),d['target_wh'])[0]+1)*d['source_wh']/2-.5;cc=np.minimum(ca,cb);cy=np.linalg.norm(back-pa,axis=1)*1600/pri[f'T{aa}']['original_wh'][0];attempts.append(dict(pair=ep.name,view='tight_CLAHE_gray' if 'tight_gray' in ep.parts else 'original_RGB_band',sampled_queries=len(pa),good_bidirectional_samples=int(((cc>=.3)&(cy<=10)).sum()),maximum_bidirectional_confidence=float(cc.max())))
 csvwrite(out/'evaluation/search_attempts.csv',attempts)
 summary=dict(position=pos,periods=4,common_T1_box_xyxy=box,RGB_common_wh=[ww,hh],point_supported_adjacent_pairs=sum(v['point_based_automatic_localization_supported'] for v in rows),total_adjacent_pairs=3,point_supported_fraction=sum(v['point_based_automatic_localization_supported'] for v in rows)/3,manual_validated_pair_fraction=None,region_localization_status='FOUR_PERIOD_CANDIDATE_ROI_WITH_UNVERIFIED_T2_T3_COARSE_LINK',review_required=True,area_percent=[v['common_area_percent'] for v in m['area_rows']],meanings=dict(geometric_common_area='Conditional on the saved observation-window transforms; not independently validated area or registration accuracy.',point_supported_pairs='Original RGB band RoMa supports a coherent point-fit similarity for T1/T2 and T3/T4. The T2/T3 ridge correlation bridge is not counted as successful point registration.',confidence='RoMa was rerun on the saved aligned original-color crops. These scores are conditional on that pose, not independent validation or calibrated probability.',manual_landmarks='None; no manual TRE is available.',root_guidance='Root-like line saliency is an algorithmic localization cue, not a verified segmentation.'),originals_byte_exact=True,RGB_nonrigid_warp=False,local_detail_box_common_xyxy=local)
 (out/'evaluation/delivery_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
