#!/usr/bin/env python3
"""Region-aware B2 scientific figures and transparent, distinct evaluation measures."""
from pathlib import Path
import json,csv,hashlib,shutil,zipfile
import cv2,numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u
ROOT=Path(__file__).resolve().parent;F=ROOT/'B2_region_revision/final';RAW=ROOT/'input/raw/V-变化/A017-1-B2';D=json.loads((F/'evaluation.json').read_text());TS=D['period_ids'];x0,y0,x1,y1=D['common_crop_xyxy'];cw,ch=x1-x0,y1-y0;w,h=D['T1_size_wh'];font=ImageFont.truetype(u.FONT,27);small=ImageFont.truetype(u.FONT,20)
# Keep native-target area statistics and unknown pixels separate from measured low confidence.
for met in D['pair_evaluation']:
 a,b=[int(z[1:]) for z in met['pair'].split('_to_')];pair=F/met['pair'];bx0,by0,bx1,by1=met['pair_T1_crop_xyxy'];source_wh=Image.open(RAW/f'T{a}.jpg').size;target_wh=Image.open(RAW/f'T{b}.jpg').size
 with h5py.File(pair/'RoMa_region_candidate_fields.h5','a') as f:
  f.attrs['source_size_wh']=source_wh;f.attrs['target_size_wh']=target_wh;f.attrs['query_canvas_size_wh']=met['pair_crop_size_wh'];conf=f['confidence_target'][:];obs=f['geometric_observable_mask'][:];points=f['confident_candidate_correspondence'][:]
 met['low_confidence_pct_pair_crop']=100*((conf<.5)&obs).mean();met['low_confidence_pct_observable_region']=100*(conf[obs]<.5).mean();met['unobservable_pct_pair_crop']=100*(~obs).mean()
 H=np.array(D['matrices_T1_to_native'][f'T{b}'])@np.array([[1,0,bx0],[0,1,by0],[0,0,1]])
 native_obs=cv2.warpPerspective(obs.astype('uint8'),H,target_wh,flags=cv2.INTER_NEAREST)>0;native_points=cv2.warpPerspective(points.astype('uint8'),H,target_wh,flags=cv2.INTER_NEAREST)>0;native_low=cv2.warpPerspective(((conf<.5)&obs).astype('uint8'),H,target_wh,flags=cv2.INTER_NEAREST)>0
 for name,mask in [('observable_mask_native_target.png',native_obs),('correspondence_support_mask_native_target.png',native_points)]:Image.fromarray(mask.astype('uint8')*255).save(pair/name)
 met.update(assessed_observable_pct_native_target=100*native_obs.mean(),point_correspondence_support_pct_native_target=100*native_points.mean(),low_confidence_pct_native_target=100*native_low.mean(),unknown_outside_assessed_pct_native_target=100*(~native_obs).mean())
 met['interpretation']+=' Local candidate displacement can also contain global-pose estimation error.'
 (pair/'evaluation.json').write_text(json.dumps(met,ensure_ascii=False,indent=2))
# Each of the six classes is shown on the SAME continuous common observation region.
classes=[('registered_RGB.png','配准原色 RGB：同一区域，保留季节变化'),('global_transform_preview.png','整体变换场预览：RGB 不做局部拉伸'),('confidence_preview.png','RoMa 置信度：点证据强弱'),('overlay_50pct.png','50%叠加：允许新生根与局部位置变化'),('checkerboard.png','棋盘格：64个T1像素'),('valid_observable_mask.png','共同可观察区域：几何有效，非逐点成功率')]
six=Image.new('RGB',(cw*2,6*(ch+62)),'white');draw=ImageDraw.Draw(six)
for k,met in enumerate(D['pair_evaluation']):
 name=met['pair'];bx0,by0,bx1,by1=met['pair_T1_crop_xyxy'];box=[x0-bx0,y0-by0,x1-bx0,y1-by0]
 for i,(filename,label) in enumerate(classes):
  im=Image.open(F/name/filename).convert('RGB').crop(box);draw.text((k*cw+12,i*(ch+62)+10),name+' '+label,font=font,fill='#26353e');six.paste(im,(k*cw,i*(ch+62)+62))
six.save(F/'six_results_region_comprehensive.png',dpi=(600,600));six.resize((1800,round(six.height*1800/six.width)),Image.Resampling.LANCZOS).save(F/'six_results_region_preview.jpg',quality=95,subsampling=0)
# Map the observation region back to every ORIGINAL canvas, proving physical region selection.
scene=Image.new('RGB',(3000,1080),'white');sd=ImageDraw.Draw(scene)
for k,t in enumerate(TS):
 im=Image.open(RAW/f'T{t}.jpg').convert('RGB');a=im.resize((1000,round(im.height*1000/im.width)),Image.Resampling.LANCZOS);scene.paste(a,(k*1000,110));H=np.array(D['matrices_T1_to_native'][f'T{t}']);corners=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),H)*1000/im.width
 points=[(float(p[0]+1000*k),float(p[1]+110)) for p in corners];sd.line(points+[points[0]],fill='#95d1dc',width=2);sd.text((k*1000+15,12),f'T{t} 原图中的共同观察区域',font=font,fill='#24353e');sd.text((k*1000+15,61),'细框定位区域；根系与土壤可发生真实变化',font=small,fill='#52646e')
scene.save(F/'region_localization_original_canvases.png',dpi=(600,600))
# Same-region / local evidence side by side, explicitly not root/new-growth classification.
qc=Image.new('RGB',(cw*3,2*(ch+85)),'white');qd=ImageDraw.Draw(qc)
for i,met in enumerate(D['pair_evaluation']):
 name=met['pair'];bx0,by0,bx1,by1=met['pair_T1_crop_xyxy'];box=[x0-bx0,y0-by0,x1-bx0,y1-by0]
 for k,(filename,label) in enumerate([('overlay_50pct.png','区域叠加：保留变化'),('confident_correspondence_mask.png','可靠点对应证据'),('local_correspondence_displacement_preview.png','候选局部位移（非生长方向）')]):
  im=Image.open(F/name/filename).convert('RGB').crop(box);qc.paste(im,(k*cw,i*(ch+85)+85));qd.text((k*cw+12,i*(ch+85)+12),name+' '+label,font=font,fill='#24343d')
qc.save(F/'region_and_local_change_evidence.png',dpi=(600,600))
# Automatically matched structure points on full T1/T2 available common view, thin markers.
m=D['pair_evaluation'][0];bx0,by0,bx1,by1=m['pair_T1_crop_xyxy'];pw,ph=m['pair_crop_size_wh'];plate=Image.new('RGB',(2*pw,ph+90),'white');pd=ImageDraw.Draw(plate)
for k,t in enumerate([1,2]):plate.paste(Image.open(F/'T1_to_T2'/f'T{t}_pair_original_RGB.png'),(k*pw,90));pd.text((k*pw+12,12),f'T{t}：稳定结构自动锚点（细线）',font=font,fill='#24343b')
rows=list(csv.DictReader((F/'T1_to_T2_all_region_candidates.csv').open()));points=[]
for row in rows:
 if row['stable_anchor']!='1':continue
 pts={1:np.array([float(row['source_x_analysis1600']),float(row['source_y_analysis1600'])])*w/1600,2:np.array([float(row['target_x_analysis1600']),float(row['target_y_analysis1600'])])*Image.open(RAW/'T2.jpg').width/1600}
 color='#b5e5e1' if row['root_support_proxy']=='1' else '#e0d9c8'
 for k,t in enumerate([1,2]):
  p=u.project(pts[t],np.linalg.inv(np.array(D['matrices_T1_to_native'][f'T{t}'])));xx,yy=p[0]-bx0+k*pw,p[1]-by0+90
  if k*pw<=xx<(k+1)*pw and 90<=yy<90+ph:pd.ellipse((xx-2,yy-2,xx+2,yy+2),outline=color,width=1)
plate.save(F/'T1-T2_stable_region_anchor_comparison.png',dpi=(600,600))
# Preview files preserve original appearance and only reduce display resolution.
for name in ['root_region_detail_comparison','common_original_RGB_montage','region_localization_original_canvases','region_and_local_change_evidence','T1-T2_stable_region_anchor_comparison']:
 im=Image.open(F/(name+'.png'));nw=min(1800,im.width);im.resize((nw,round(im.height*nw/im.width)),Image.Resampling.LANCZOS).save(F/(name+'_preview.jpg'),quality=96,subsampling=0)
# Root-window quality is described as point evidence, never a gate on observational content.
window_rows=[]
for win in D['observation_windows']:
 for met in D['pair_evaluation']:
  bx0,by0,bx1,by1=met['pair_T1_crop_xyxy'];wx0,wy0,wx1,wy1=win['T1_xyxy'];sl=np.s_[wy0-by0:wy1-by0,wx0-bx0:wx1-bx0]
  with h5py.File(F/met['pair']/'RoMa_region_candidate_fields.h5','r') as f:
   ok=f['confident_candidate_correspondence'][sl];conf=f['confidence_target'][sl];shift=f['local_displacement_target_to_source_T1_px'][sl];obs=f['geometric_observable_mask'][sl]
  motion=np.linalg.norm(shift,axis=-1);window_rows.append(dict(pair=met['pair'],window=win['label'],description=win['description'],observable_pct=100*obs.mean(),correspondence_support_pct=100*ok.mean(),locally_displaced_support_pct=100*(ok&(motion>3)).mean(),low_confidence_pct=100*(conf<.5).mean(),interpretation='Low support is not invalid observational content or proof of new/dead roots.'))
(F/'root_window_evidence.json').write_text(json.dumps(window_rows,ensure_ascii=False,indent=2))
D['input_sha256']={f'T{t}':hashlib.sha256((RAW/f'T{t}.jpg').read_bytes()).hexdigest() for t in TS};(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
# Confirm lossless PNG pixel values and isotropic transformations, not merely file existence.
with h5py.File(F/'global_region_coordinate_fields.h5','r') as f:
 for t in TS:
  H=f[f'T{t}/T1_to_native_matrix'][:];s=np.linalg.svd(H[:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10)
  coords=f[f'T{t}/common_T1_grid_to_native_xy'][:];source=np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB'));np.testing.assert_array_equal(u.remap(source,coords),np.array(Image.open(F/f'T{t}_common_original_RGB.png')))
print('PASS faithful original RGB and shape-preserving transforms',flush=True)
lines=['# B2 同一区域连续配准修订','', 'T1为苗期；实际序列T1→T2→T4（T3缺失）。目标是定位同一微根管观察区域，允许新生根、延伸、局部位移和土壤变化，不要求所有根段与土壤纹理相同。','',f'三期共同最大横向裁剪：{cw}×{ch}个T1像素；共同几何可观察面积占完整T1 {D["common_observable_pct_T1_full"]:.2f}%。完整原色共同视野与逐点对应证据分别保存。', '', '| 对应时期 | 稳定结构锚点 | 根形支持候选 | 自动验证中位误差/T1像素 | 同一区域定位支持 | 局部点对应支持/该对视野 | 低置信度/该对视野 |', '|---|---:|---:|---:|---|---:|---:|']
for met in D['pair_evaluation']:
 fs=next(z for z in D['region_features'] if z['pair']==met['pair']);lines.append(f'| {met["pair"]} | {fs["stable_structure_anchors"]} | {fs["root_support_anchor_candidates"]} | {fs["spatial_cv_anchor_median_T1_px"]:.2f} | {"支持" if met["region_localization_supported"] else "不足"} | {met["point_correspondence_support_pct_pair_crop"]:.2f}% | {met["low_confidence_pct_pair_crop"]:.2f}% |')
lines+=['', f'区域定位自动规则通过比例：{D["region_localization_supported_pair_ratio"]:.0%}；这衡量同一区域定位证据，不是逐条根追踪成功率。链式与独立T1→T4定位中位差 {D["chain_vs_direct_median_T1_px"]:.2f}个T1像素。', '',f'三期共同点对应支持占共同裁剪 {D["common_point_correspondence_support_pct_crop"]:.2f}%，占完整T1 {D["common_point_correspondence_support_pct_T1_full"]:.2f}%。该面积用于展示模型证据，不用于裁掉新生根或判定整个区域定位失败。','', '之前的4.98%与1/2通过是严格同位置点对应口径，不能作为共同区域大小或同一区域定位失败率。保留旧指标和版本供核对。本修订真实扩大完整原图、多尺度结构搜索和T1→T2共同视野，并重新运行RoMa上下文窗口。', '', '共同可观察掩膜：原图坐标映射的几何有效范围。可靠点对应掩膜：RoMa双向置信度≥0.5、循环误差≤3个T1像素、候选场正Jacobian；取消与全局位置偏差≤3像素的限制，允许局部位移，未调整模型置信度。局部位移箭头表示目标查询点到候选源点的方向，不是根系生长方向，也不是确认的土层迁移。','', '区域定位规则：至少20个稳定自动锚点；共识锚点内空间交叉验证中位误差≤3个T1像素；锚点横向跨度≥该对视野40%，纵向跨度≥20%；链式/独立定位中位差≤5个T1像素。评价方式因用户明确目标为区域定位而调整，不能与旧逐点规则混为一个通过率。', '', '自动验证在描述符共识锚点内进行，没有独立人工对应坐标；真实标志点TRE未评估。新生/死亡未自动分类，低置信度也可能来自遮挡、照明差异或模型限制。', '', 'RGB只作整体等比缩放、旋转、平移，原色只采样一次，不用RoMa局部场拉伸图像。T2原宽1330像素，统一尺度与600 DPI不会增加未采集的细节。', '', '全图存在多个相似条带；目标条带依据用户提示的根段/分叉组合与连续定位核对，其他候选保存在完整搜索JSON中。']
(F/'result_summary.md').write_text('\n'.join(lines)+'\n')
scripts=F/'scripts';scripts.mkdir(exist_ok=True)
for name in ['search_B2_region.py','B2_region_tools.py','build_B2_region_revision.py','package_B2_region_revision.py','publish_B2_region_revision.py','run_0353_1_root_priority.py','register_roma.py','evaluate_adjacent_registration.py']:shutil.copy2(ROOT/name,scripts/name)
shutil.copy2(ROOT/'B2_region_revision/full_canvas_feature_search.json',F/'full_canvas_feature_search.json')
(scripts/'REPRODUCE.md').write_text('在已配置RoMa的LI环境，将本目录Python文件放到results/microtube_registration；原图放在input/raw/V-变化/A017-1-B2。先运行search_B2_region.py，再build_B2_region_revision.py与package_B2_region_revision.py。脚本自动检测CUDA。build还引用保留的B2_shape_revision/final/evaluation.json及shape_feature_candidates.json作为用户所指区域的初始定位；这些文件在仓库上一修订及本结果的reference_prior中保留。publish是已有上传工作区的可选发布脚本。所有其他旧helper只导入函数，不运行其主程序。\n')
prior=F/'reference_prior';prior.mkdir(exist_ok=True);oldp=ROOT/'B2_shape_revision/final/evaluation.json';oldp=oldp if oldp.exists() else ROOT.parent/'reference_prior/previous_evaluation.json';oldc=ROOT/'B2_shape_revision/shape_feature_candidates.json';oldc=oldc if oldc.exists() else ROOT.parent/'reference_prior/previous_shape_feature_candidates.json';shutil.copy2(oldp,prior/'previous_evaluation.json');shutil.copy2(oldc,prior/'previous_shape_feature_candidates.json')
P=Path('/workspace/LI/results');archives={
 'A017-1-B2_region_results.zip':[p for p in F.rglob('*') if p.is_file() and p.suffix!='.h5' and not any(q in ['T1_to_T2','T2_to_T4'] for q in p.relative_to(F).parts) and '_full_T1_RGB' not in p.name and 'literal_native' not in p.name],
 'A017-1-B2_region_T1-T2_images.zip':[p for p in (F/'T1_to_T2').glob('*') if p.is_file() and p.suffix!='.h5'],
 'A017-1-B2_region_T2-T4_images.zip':[p for p in (F/'T2_to_T4').glob('*') if p.is_file() and p.suffix!='.h5'],
 'A017-1-B2_region_original_RGB.zip':list(F.glob('*_full_T1_RGB.png'))+list(F.glob('*literal_native*'))+[F/'common_observable_mask_T1.png'],
 'A017-1-B2_region_global_fields.zip':[F/'global_region_coordinate_fields.h5'],
 'A017-1-B2_region_T1-T2_RoMa_fields.zip':[F/'T1_to_T2/RoMa_region_candidate_fields.h5'],
 'A017-1-B2_region_T2-T4_RoMa_fields.zip':[F/'T2_to_T4/RoMa_region_candidate_fields.h5']}
for name,files in archives.items():
 with zipfile.ZipFile(P/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(F))
 with zipfile.ZipFile(P/name) as z:assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()))
 assert (P/name).stat().st_size<100*1024**2,(name,(P/name).stat().st_size);print('REGION_ARCHIVE',name,round((P/name).stat().st_size/1024**2,2),'MiB',flush=True)
print('FINAL_REGION_EVALUATION',json.dumps({k:D[k] for k in ['common_crop_size_wh','common_observable_pct_T1_full','region_localization_supported_pair_ratio','common_point_correspondence_support_pct_crop','pair_evaluation']},ensure_ascii=False),flush=True)
