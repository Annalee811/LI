"""Save only evidence-supported geometry, retaining unlocated T1 separately."""
import os,json,hashlib,csv,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
os.environ.setdefault('MPLCONFIGDIR','/tmp/LI_Q1_recheck_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(os.environ.get('LI_Q1_RECHECK_WORK','/tmp/LI_Q1_3_rootguided_v3'));raw=Path(os.environ.get('LI_Q1_RECHECK_RAW','/tmp/LI_all_series_localization_v2/inputs/raw/Q1-3-Z3'));out=r/'deliverable';out.mkdir(exist_ok=True);cv2.setNumThreads(1)
for k in ['originals','RGB','localization','QC','fields','confidence','masks','evaluation']: (out/k).mkdir(exist_ok=True)
pri=json.loads((r/'search_priors.json').read_text());boxes={int(k):v for k,v in pri['native_boxes'].items()};ims={t:np.array(Image.open(next(raw.glob(f'T{t}.*'))).convert('RGB')) for t in [1,2,3]};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={}
for t in ims:
 p=next(raw.glob(f'T{t}.*'));shutil.copy2(p,out/'originals'/p.name);hashes[f'T{t}']=sha(p);x0,y0,x1,y1=boxes[t];Image.fromarray(ims[t][y0:y1,x0:x1]).save(out/'RGB'/f'T{t}_user_guided_search_ROI_original_RGB.png',dpi=(600,600))
# Choose least restrictive fixed root-ridge threshold; never select by held-out residual.
fit=json.loads((r/'root_pose_candidates.json').read_text())[0];assert fit['root_ridge_threshold']==.1 and fit['anchors']>=20;H=np.array(fit['matrix_native']);S=np.linalg.svd(H[:2,:2],compute_uv=False);assert np.isclose(*S,rtol=1e-10)
y,x=np.mgrid[:ims[2].shape[0],:ims[2].shape[1]].astype('float32');q=np.stack([x,y],-1);target=q@H[:2,:2].T+H[:2,2];b2=boxes[2];b3=boxes[3];valid=(x>=b2[0])&(x<b2[2])&(y>=b2[1])&(y<b2[3])&(target[...,0]>=b3[0])&(target[...,0]<=b3[2]-1)&(target[...,1]>=b3[1])&(target[...,1]<=b3[3]-1)
# Maximum horizontal rectangle within both user-guided search bands and image bounds.
heights=np.zeros(valid.shape[1],int);best=(0,0,0,0,0)
for yy,row in enumerate(valid):
 heights=np.where(row,heights+1,0);stack=[]
 for xx in range(len(heights)+1):
  hh=int(heights[xx]) if xx<len(heights) else 0;start=xx
  while stack and stack[-1][1]>hh:
   sx,hv=stack.pop();ww=xx-sx;start=sx
   if ww>=hv and ww*hv>best[0]:best=(ww*hv,sx,yy-hv+1,xx,yy+1)
  if not stack or stack[-1][1]<hh:stack.append((start,hh))
_,x0,y0,x1,y1=best;assert best[0]>100000;box=[x0,y0,x1,y1];q=q[y0:y1,x0:x1];target=target[y0:y1,x0:x1].astype('float32');A=ims[2][y0:y1,x0:x1];B=cv2.remap(ims[3],target[...,0],target[...,1],cv2.INTER_LINEAR);h,w=A.shape[:2]
Image.fromarray(A).save(out/'RGB'/'T2_confirmed_common_RGB.png',dpi=(600,600));Image.fromarray(B).save(out/'RGB'/'T3_aligned_to_T2_common_RGB.png',dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(out/'RGB'/'T2_confirmed_common_RGB.png')),A)
# Remap from full native coordinates into the cropped RoMa input grid.
posed=(r/'posed_T2_to_T3/dense.npz').exists();z=np.load(r/('posed_T2_to_T3/dense.npz' if posed else 'T2_to_T3/dense.npz'));d={k:z[k].astype('float32') for k in z.files};pa=q-np.array(box[:2] if posed else b2[:2]);wa=d['source_wh'];wb=d['target_wh']
def sample(a,p,wh):
 hh,ww=a.shape[:2];s=(p+.5)*[ww/wh[0],hh/wh[1]]-.5
 return cv2.remap(a,s[...,0].astype('float32'),s[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
pt=(sample(d['forward'],pa,wa)+1)*[wb[0]/2,wb[1]/2]-.5;ca=sample(d['ca'],pa,wa);cb=sample(d['cb'],pt,wb);back=(sample(d['backward'],pt,wb)+1)*[wa[0]/2,wa[1]/2]-.5;cf=np.minimum(ca,cb).clip(0,1);pred_native=(pt+np.array(box[:2]))@H[:2,:2].T+H[:2,2] if posed else pt+np.array(b3[:2]);cycle=np.linalg.norm(back-pa,axis=-1);discrepancy=np.linalg.norm(pred_native-target,axis=-1);owner=(pt[...,0]>=0)&(pt[...,0]<=wb[0]-1)&(pt[...,1]>=0)&(pt[...,1]<=wb[1]-1);cf*=owner;effective=owner&(cf>=.5)&(cycle<=10)&(discrepancy<=15)
np.savez_compressed(out/'fields'/'T2_to_T3_RoMa_native_fields_float32.npz',query_xy_native_T2=q,dense_target_xy_native_T3=pred_native.astype('float32'),dense_displacement_xy=(pred_native-q).astype('float32'),similarity_target_xy_native_T3=target,similarity_displacement_xy=target-q,cycle_error_T2_px=cycle.astype('float32'),model_disagreement_T3_px=discrepancy.astype('float32'))
np.savez_compressed(out/'confidence'/'T2_to_T3_confidence_float32.npz',confidence=cf.astype('float32'));Image.fromarray(np.uint16(np.rint(cf*65535))).save(out/'confidence'/'T2_to_T3_confidence_16bit.png');Image.fromarray(np.uint8(effective)*255).save(out/'masks'/'T2_to_T3_confidence_filtered_correspondence_mask.png');Image.fromarray(np.full((h,w),255,np.uint8)).save(out/'masks'/'T2_T3_geometric_common_mask.png');Image.fromarray(np.uint8(cf<.5)*255).save(out/'masks'/'T2_to_T3_low_confidence_mask.png')
polys={2:np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32)};polys[3]=polys[2]@H[:2,:2].T+H[:2,2];area=w*h;rows=[]
for t in [1,2,3]:
 native=np.zeros(ims[t].shape[:2],np.uint8)
 if t>1:cv2.fillConvexPoly(native,np.rint(polys[t]*16).astype('int32'),255,shift=4);Image.fromarray(native).save(out/'masks'/f'T{t}_native_T2_T3_geometric_mask.png')
 rows.append(dict(period=f'T{t}',original_width=ims[t].shape[1],original_height=ims[t].shape[0],original_area_px2=ims[t].shape[0]*ims[t].shape[1],T1_T2_T3_common_area_px2=None,T1_T2_T3_common_area_percent=None,T2_T3_pair_native_common_area_px2=area*(np.linalg.det(H[:2,:2]) if t==3 else 1) if t>1 else None,T2_T3_pair_area_percent=100*area*(np.linalg.det(H[:2,:2]) if t==3 else 1)/(ims[t].shape[0]*ims[t].shape[1]) if t>1 else None,status='T1_connection_unresolved' if t==1 else 'T2_T3_pair_only'))
with (out/'evaluation/area_ratios.csv').open('w',encoding='utf-8-sig',newline='') as f:
 wr=csv.DictWriter(f,list(rows[0]));wr.writeheader();wr.writerows(rows)
# Six requested QC categories, all using the same unaltered-colour pair crop.
over=np.rint((A.astype(float)+B)/2).astype('uint8');yy,xx=np.indices((h,w));check=np.where((((xx//48+yy//48)%2)==0)[...,None],A,B);cimg=np.uint8(np.rint(plt.get_cmap('cividis')(cf)[...,:3]*255));mask=np.uint8(effective)*255
for name,im in [('aligned_T3_in_T2',B),('overlay_50pct',over),('checkerboard',check),('confidence_preview',cimg),('effective_correspondence_mask',mask)]:Image.fromarray(im).save(out/'QC'/f'{name}.png',dpi=(600,600))
fig,ax=plt.subplots(figsize=(8,3));ax.imshow(A);step=60;ax.quiver(xx[::step,::step],yy[::step,::step],(pred_native-q)[::step,::step,0],(pred_native-q)[::step,::step,1],cf[::step,::step],cmap='cividis',clim=(0,1),angles='xy',scale_units='xy',scale=12,width=.003);ax.set_title('Dense displacement T2 → T3 (arrows ÷ 12)');ax.axis('off');fig.savefig(out/'QC'/'transform_field_preview.png',dpi=300,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained');fig.suptitle('Q1-3-Z3 | root-guided T2 → T3 | T1 connection unconfirmed',fontsize=14)
for ax,title,im in zip(axs.flat,['Aligned T3 in T2 coordinates','Dense transform preview','Pose-conditioned RoMa confidence','50% RGB overlay','Checkerboard QC','Confidence-filtered correspondence mask'],[B,np.array(Image.open(out/'QC/transform_field_preview.png')),cimg,over,check,mask]):ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=11);ax.axis('off')
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='cividis'),ax=axs[0,2],fraction=.025,pad=.02,label='Confidence')
fig.savefig(out/'QC'/'Q1-3-Z3_six_category_overview.png',dpi=300);fig.savefig(out/'QC'/'Q1-3-Z3_six_category_overview.pdf');plt.close(fig)
# Original views, showing the supplied region prior separately from the confirmed polygon.
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=lambda n:ImageFont.truetype(fontpath,n);pw=800;ph=round(2273*pw/2550);canvas=Image.new('RGB',(pw*3,ph+150),'white');draw=ImageDraw.Draw(canvas)
for i,t in enumerate([1,2,3]):
 im=Image.fromarray(ims[t]);im=im.resize((pw,round(im.height*pw/im.width)),Image.Resampling.LANCZOS);canvas.paste(im,(i*pw,80));sc=pw/ims[t].shape[1];b=boxes[t];draw.rectangle((i*pw+b[0]*sc,80+b[1]*sc,i*pw+b[2]*sc,80+b[3]*sc),outline='#D7903B',width=2)
 if t>1:
  pp=[tuple(v) for v in polys[t]*sc+[i*pw,80]];draw.line(pp+[pp[0]],fill='#1C9A95',width=3)
 draw.text((i*pw+14,8),f'Q1-3-Z3 | T{t}',font=font(29),fill='#253A43');draw.text((i*pw+14,45),'T1：框选搜索区，连接未确认' if t==1 else '青色：已建立 T2/T3 对应区',font=font(24),fill='#A66F2D' if t==1 else '#177F7A')
draw.text((18,ph+95),'橙框：依据用户指引的搜索范围；T1 尚无可靠自动对应，不代表三期精确配准。T4 缺失。',font=font(24),fill='#354A52');canvas.save(out/'localization'/'Q1-3-Z3_rechecked_localization_overview.png',dpi=(600,600));canvas.save(out/'localization'/'Q1-3-Z3_rechecked_localization_overview.jpg',quality=95)
# Audit exact failures, not falsely empty masks or zero area for an unknown link.
audit=[]
for prefix in ['','enhanced_']:
 for a,b in ([(1,2),(1,3),(2,3)] if not prefix else [(1,2),(1,3)]):
  z=np.load(r/f'{prefix}T{a}_to_T{b}/matches.npz');c=z['confidence'];cy=z['cycle_analysis1600'];audit.append(dict(pair=f'T{a}_to_T{b}',matching_view='CLAHE_grayscale' if prefix else 'original_RGB_crop',sampled_queries=len(c),median_bidirectional_confidence=float(np.median(c)),maximum_bidirectional_confidence=float(c.max()),bidirectional_c_ge_0_3_cycle_le_10_analysis1600=int(((c>=.3)&(cy<=10)).sum())))
T1_unknown=dict(status='unresolved',reason='No reliable bidirectional T1 links in either RGB-region or CLAHE-region RoMa checks; source region guidance is not point correspondence ground truth.',T1_to_T2_transform=None,T1_to_T3_cumulative_transform=None,T1_cumulative_confidence=None,three_period_common_area=None);(out/'confidence/T1_referenced_cumulative_UNAVAILABLE.json').write_text(json.dumps(T1_unknown,indent=2))
metadata=dict(position='Q1-3-Z3',input_sha256=hashes,periods=[1,2,3],missing_periods=[4],T1_link=T1_unknown,T2_to_T3_root_similarity=fit,T2_reference_box=box,RGB_output_wh=[w,h],source_search_priors=pri,scope='Only Q1-3-Z3 is updated; all other locations and the previous full-series archive are unchanged.',root_ridge_note='Frangi saliency prioritizes elongated structures; it is not a biological root segmentation ground truth.',validation_note='Descriptor points were not used for this RoMa refit, but were selected by an earlier consensus. These are conditional automatic errors, not manual TRE or external validation.',RGB_rendering='T2 is an exact native crop. T3 is sampled once with a global uniform similarity. Original colors preserved; no elastic deformation. T1 saved as an unaligned user-guided native crop only.',search_audit=audit,pair_QC=dict(input_view='fixed aligned original-color RGB' if posed else 'search region original-color RGB',conditional_on_saved_similarity=posed,not_independent_localization_validation=True,confidence_threshold=.5,cycle_limit_T2_px=10,RoMa_vs_similarity_limit_T3_px=15,effective_pixels=int(effective.sum()),common_geometric_pixels=area,effective_percent=100*effective.mean(),low_confidence_percent=100*(cf<.5).mean(),mean_confidence=float(cf.mean())),adjacent_pair_localization=dict(available_pairs=2,supported_pairs=1,fraction=.5,denominator='T1→T2 and T2→T3; region/correspondence evidence only, not manual accuracy'))
(out/'evaluation/recheck_audit.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2));(out/'evaluation/area_ratios.json').write_text(json.dumps(rows,indent=2));(out/'fields/T2_to_T3_similarity.json').write_text(json.dumps(fit,indent=2));shutil.copy2(r/'descriptor_probe.json',out/'evaluation/descriptor_probe.json');shutil.copy2(r/'root_pose_candidates.json',out/'evaluation/root_pose_candidates.json')
assert all(sha(out/'originals'/next(raw.glob(f'T{t}.*')).name)==hashes[f'T{t}'] for t in ims);np.testing.assert_array_equal(np.array(Image.open(out/'RGB'/'T3_aligned_to_T2_common_RGB.png')),B)
(out/'README.md').write_text('''# Q1-3-Z3 根系优先重新核查\n\n仅更新此位置。橙框为按用户指引设定的宽松搜索区，青框为自动证据支持的 T2/T3 共视区。\n\nT2/T3 采用原色裁剪和单次全局等比例变换；不做弹性变形。T1 搜索区原色裁剪独立保留，但尚不能与 T2/T3 建立可靠对应。T4 缺失。不要把三张搜索区裁剪当作已配准序列。\n\n有效共视掩膜表示几何可观察区域；置信度过滤掩膜表示满足给定阈值的对应质控像素，均不是根系分割掩膜。T2→T3 只有一个有效边，置信度即该单步置信度；最终质控在已经固定的全局定位裁剪后重新运行 RoMa，以该定位为条件，不是定位正确性的独立证明。T1 起点累计置信度、三期共区和比例均未确定，保留 null，未伪造为零。\n\n评估中根状结构匹配点用于拟合，另列先前共识选出的自动描述子核对误差；没有人工标志点，不是人工 TRE。\n\n完整脚本见 scripts/reproduce.sh；原图哈希、模型哈希、阈值与核查记录一并保留。\n''',encoding='utf-8')
fig,axs=plt.subplots(2,2,figsize=(12,6),layout='constrained')
for col,(t,im) in enumerate([(2,Image.fromarray(A)),(3,Image.fromarray(B))]):
 axs[0,col].imshow(im);axs[0,col].set_title(f'T{t} | common region');axs[1,col].imshow(im.crop((520,170,900,449)));axs[1,col].set_title(f'T{t} | root structure detail')
for ax in axs.flat:ax.axis('off')
fig.suptitle('Q1-3-Z3 | T2/T3 root-guided alignment | original colors');fig.savefig(out/'QC/T2_T3_full_and_root_detail.png',dpi=300);fig.savefig(out/'QC/T2_T3_full_and_root_detail.pdf');plt.close(fig)
print(json.dumps(dict(box=box,RGB_wh=[w,h],area_rows=rows,pair_QC=metadata['pair_QC'],search_audit=audit),ensure_ascii=False,indent=2))
