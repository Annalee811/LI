from pathlib import Path
import csv,json,hashlib,base64
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parents[2];KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation';OUT=ROOT/'results/B3_AI_candidate_review'
rows=list(csv.DictReader((KIT/'split_manifest.csv').open(encoding='utf-8-sig')))
traces={};window=None
for line in Path(__file__).with_name('visual_trace_seeds.txt').read_text().splitlines():
 if not line or line.startswith('#'):continue
 if line.startswith('['):window=line[1:-1];traces[window]=[];continue
 w,p=line.split(':');traces[window].append((float(w),np.array([[float(v) for v in q.split(',')] for q in p.split()],np.float32)))
for d in ['candidate_masks','attributes','overlays','roots_only','review_panels','metadata','originals']:(OUT/d).mkdir(exist_ok=True)
PALETTE={0:(37,99,235),1:(22,163,74),2:(234,179,8),3:(147,51,234)}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
def sample(im,p):
 p=np.asarray(p,np.float32);return cv2.remap(im,p[:,0].reshape(-1,1),p[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT).reshape((len(p),)+im.shape[2:])
def dense(p):
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];s=np.arange(0,max(arc[-1],1),.75);return np.column_stack([np.interp(s,arc,p[:,i]) for i in range(2)]).astype(np.float32)
def fit(im,p,cap,search=15):
 gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32);blur=cv2.GaussianBlur(gray,(0,0),.65)
 grad=np.gradient(p,axis=0);grad/=np.maximum(np.linalg.norm(grad,axis=1,keepdims=True),1e-5);normal=np.column_stack([-grad[:,1],grad[:,0]])
 offsets=np.arange(-search,search+.01,.5,dtype=np.float32);profile=np.stack([sample(blur,p+normal*o) for o in offsets],axis=1)
 # Centre ridges must exceed both shoulders. This is local fitting inside visually selected corridors, not a trained root classifier.
 left=np.stack([sample(blur,p+normal*(o-4)) for o in offsets],axis=1)
 right=np.stack([sample(blur,p+normal*(o+4)) for o in offsets],axis=1)
 response=np.minimum(profile-left,profile-right)
 score=response-.28*np.abs(offsets)[None,:]
 n,m=score.shape;dp=score[0].copy();back=np.zeros((n,m),np.int16);penalty=1.25*(offsets[:,None]-offsets[None,:])**2
 for k in range(1,n):
  v=dp[:,None]-penalty;back[k]=np.argmax(v,axis=0);dp=v.max(axis=0)+score[k]
 choice=np.zeros(n,np.int16);choice[-1]=np.argmax(dp)
 for k in range(n-1,0,-1):choice[k-1]=back[k,choice[k]]
 shift=gaussian_filter1d(offsets[choice],3);cent=p+normal*shift[:,None]
 center=sample(blur,cent);side1=sample(blur,cent+normal*4);side2=sample(blur,cent-normal*4);bg=(side1+side2)/2;contrast=np.minimum(center-side1,center-side2)
 across=np.arange(-4,4.01,.25);prof=np.stack([sample(blur,cent+normal*o) for o in across],axis=1)
 half=bg[:,None]+np.maximum(contrast,0)[:,None]*.5;valid=prof>=half
 widths=[]
 for v in valid:
  c=len(across)//2;lo=hi=c
  while lo>0 and v[lo-1]:lo-=1
  while hi<len(across)-1 and v[hi+1]:hi+=1
  widths.append(np.clip((hi-lo+1)*.25,.85,cap))
 widths=gaussian_filter1d(np.array(widths),2)
 # Keep low-contrast and discontinuous intervals unresolved rather than drawing through hidden roots.
 support=(contrast>1.65)&(np.abs(shift)<search-.2)
 support=gaussian_filter1d(support.astype(float),1.3)>.65
 lap=cv2.Laplacian(gray,cv2.CV_32F);localvar=cv2.GaussianBlur(lap*lap,(0,0),3)
 glare=(sample(localvar,cent)>150)&(sample(gray,cent)>85)
 # Preserve bright roots; reflection ambiguity is retained separately for temporal decisions.
 return cent,widths,contrast,support,glare
records=[];audit=[];instance_rows=[]
for wi,(win,paths) in enumerate(traces.items()):
 images={int(r['period']):np.array(Image.open(KIT/r['image_path']).convert('RGB')) for r in rows if r['window_id']==win}
 ref=images[3];norm=lambda im:cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))
 flows={3:np.zeros((448,448,2),np.float32)}
 for t in [1,2,4]:
  f=cv2.calcOpticalFlowFarneback(norm(ref),norm(images[t]),None,.5,4,31,4,7,1.5,0);f=cv2.GaussianBlur(f,(0,0),2);scale=np.minimum(1,10/np.maximum(np.linalg.norm(f,axis=2),1e-5));flows[t]=f*scale[:,:,None]
 fits={t:[] for t in range(1,5)}
 for j,(cap,seed) in enumerate(paths):
  p=dense(seed)
  ref_fit=fit(images[3],p,cap)
  fits[3].append(ref_fit)
  reference=ref_fit[0]
  for t in [1,2,4]:
   # Use the visually guided T3 root shape; only allow small local correction at other dates.
   displacement=sample(flows[t],reference);displacement=gaussian_filter1d(displacement,8,axis=0)
   fitted=fit(images[t],reference+displacement,cap,search=4)
   fits[t].append(fitted)
 panel=Image.new('RGB',(960,1010),'white');pd=ImageDraw.Draw(panel)
 for t in range(1,5):
  row=next(r for r in rows if r['window_id']==win and int(r['period'])==t);id=row['image_id'];im=images[t];mask=np.full((448,448),64,np.uint8);attrs={k:np.zeros_like(mask) for k in ['color','temporal','viability']};instances=np.zeros((448,448),np.uint16)
  for j,(cent,widths,contrast,support,glare) in enumerate(fits[t]):
   previous=fits[t-1][j][3] if t>1 else np.zeros_like(support);baseline=fits[1][j][3]
   existing=support & (previous|baseline) if t>1 else np.zeros_like(support)
   temporal=np.zeros(len(support),np.uint8);temporal[existing]=3
   # New/growing are proposals only; never infer either from a hidden previous interval.
   if t>1:
    prev_glare=fits[t-1][j][4];prior_fraction=previous.mean();current_fraction=support.mean()
    if prior_fraction<.08 and current_fraction>.7 and prev_glare.mean()<.06:
     temporal[support]=1
    elif prior_fraction>.45:
     inds=np.flatnonzero(previous)
     if len(inds):
      lo,hi=inds[0],inds[-1];newend=((np.arange(len(support))<lo-6)|(np.arange(len(support))>hi+6))&support&~prev_glare
      temporal[newend]=2
   counts={str(v):int(np.count_nonzero(temporal[support]==v)) for v in range(4)}
   activewidths=widths[support];track=f'{win}_trace_{j+1:03}'
   instance_rows.append(dict(image_id=id,provisional_trace_id=track,reference_period='T3',visible_samples=int(support.sum()),total_samples=len(support),candidate_visible_fraction=round(float(support.mean()),4),width_median_px=round(float(np.median(activewidths)),2) if len(activewidths) else None,width_95pct_px=round(float(np.percentile(activewidths,95)),2) if len(activewidths) else None,temporal_sample_counts=counts,death_status='not_determined',independent_human_review=False))
   for k,((x,y),w,ok,cl) in enumerate(zip(cent,widths,support,temporal)):
    xx,yy=int(round(x)),int(round(y))
    if not(0<=xx<448 and 0<=yy<448):continue
    r=max(.45,float(w)/2);x0=max(0,int(np.floor(x-r)));x1=min(447,int(np.ceil(x+r)));y0=max(0,int(np.floor(y-r)));y1=min(447,int(np.ceil(y+r)))
    for py in range(y0,y1+1):
     for px in range(x0,x1+1):
      if (px-x)**2+(py-y)**2>r*r:continue
      if ok:
       mask[py,px]=255;attrs['temporal'][py,px]=cl;instances[py,px]=j+1
       # Colour is left unclassified under warm lighting or low contrast.
       rgb=im[py,px].astype(float);attrs['color'][py,px]=1 if contrast[k]>6 and rgb.min()/max(rgb.max(),1)>.67 else 0
      elif mask[py,px]!=255:mask[py,px]=128
  Image.fromarray(mask).save(OUT/'candidate_masks'/(id+'.png'));Image.fromarray(instances).save(OUT/'attributes'/(id+'_trace_ids.png'))
  for name,a in attrs.items():Image.fromarray(a).save(OUT/'attributes'/(id+'_'+name+'.png'))
  colored=np.zeros_like(im)
  for value,rgb in PALETTE.items():colored[(mask==255)&(attrs['temporal']==value)]=rgb
  overlay=im.copy();root=mask==255;overlay[root]=(im[root]*.25+colored[root]*.75).astype(np.uint8)
  # Grey dotted corridors expose locations needing review without obscuring the original photograph.
  yy,xx=np.indices(mask.shape);unc=(mask==128)&((xx+yy)%5==0);overlay[unc]=(150,163,184)
  Image.fromarray(overlay).save(OUT/'overlays'/(id+'_overlay.png'))
  rgba=np.dstack([colored,np.where(root,255,0).astype(np.uint8)]);Image.fromarray(rgba).save(OUT/'roots_only'/(id+'_roots.png'))
  Image.fromarray(im).save(OUT/'originals'/(id+'.png'))
  # Preserve input PNG bytes exactly, not re-encoded equivalents.
  (OUT/'originals'/(id+'.png')).write_bytes((KIT/row['image_path']).read_bytes())
  def enc(a):return base64.b64encode(a.tobytes()).decode()
  record=dict(name=id+'.png',image_sha256=row['image_sha256'],mask=enc(mask),color=enc(attrs['color']),temporal=enc(attrs['temporal']),viability=enc(attrs['viability']),reviewed=False,evidence='',annotator='',prefill_origin='assistant_visual_T3_corridor_seeds_plus_automated_temporal_flow_and_local_profile_fit',modified=False)
  records.append(record)
  meta=dict(image_id=id,image_sha256=row['image_sha256'],mask_sha256=hashlib.sha256((OUT/'candidate_masks'/(id+'.png')).read_bytes()).hexdigest(),source_period='T'+str(t),status='pending_person_review',label_origin=record['prefill_origin'],reviewed_entire_image=False,not_training_ground_truth=True,label_encoding=dict(unreviewed=64,root_candidate=255,uncertain=128,background=0),no_confirmed_death_from_rgb=True,unclassified_colour_is_not_no_browning=True,trace_ids_are_provisional_not_verified_biological_roots=True)
  (OUT/'metadata'/(id+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
  audit.append(dict(image_id=id,split=row['split'],root_candidate_pixels=int(root.sum()),uncertain_pixels=int((mask==128).sum()),unreviewed_pixels=int((mask==64).sum()),root_candidate_area_fraction=round(float(root.mean()),5),human_reviewed=False))
  ox=16+480*((t-1)%2);oy=45+500*((t-1)//2);pd.text((ox,oy-27),id+' | pending review',fill='black',font=font);panel.paste(Image.fromarray(overlay),(ox,oy))
 panel.save(OUT/'review_panels'/(win+'_four_period_review.png'))
 print(win,'4_PERIOD_PREFILLS_READY',flush=True)
(OUT/'embedded_prefills.json').write_text(json.dumps(records,separators=(',',':')))
(OUT/'candidate_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'trace_measurements.json').write_text(json.dumps(instance_rows,indent=2)+'\n')
print('ALL_60_PENDING_REVIEW',len(records),flush=True)
