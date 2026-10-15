from pathlib import Path
import base64,csv,hashlib,json
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize,remove_small_objects
ROOT=Path(__file__).resolve().parents[2];KIT=ROOT/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation';OLD=KIT/'reviewable-annotations';OUT=ROOT/'results/B3_reannotation_v2'
rows=list(csv.DictReader((KIT/'split_manifest.csv').open(encoding='utf-8-sig')));reference=json.loads(Path(__file__).with_name('example_traces_v2.json').read_text());S=4
OUT.mkdir(parents=True,exist_ok=True)
for name in ['masks','attributes','overlays','roots_only','review_panels','originals','metadata']:(OUT/name).mkdir(exist_ok=True)
for d in ['roots','scene']:(OUT/'overlays'/d).mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
def line_mask(points,width):
 a=Image.new('L',(448*S,448*S));d=ImageDraw.Draw(a);d.line([(round(x*S),round(y*S)) for x,y in points],fill=255,width=max(1,round(width*S)),joint='curve');return np.array(a.resize((448,448),Image.Resampling.LANCZOS))>=100
# Manually specified wet zones restrict droplet proposals; circles do not serve as root cues.
def wet_zone(window,t):
 y,x=np.indices((448,448));z=np.zeros((448,448),bool)
 if t not in [1,2]:return z
 if window.startswith('B3_A0_C0'):z=((y<140)&(x>220))|((y>298)&(x<330))|((x>370)&(y<290))
 elif window.startswith(('B3_A0_C1','B3_A0_C2','B3_A0_C3','B3_A1_C2','B3_A1_C3')):z[:]=True
 elif window=='B3_A0_C4':z=(y>190)&(x<180)
 elif window=='B3_A1_C1':z=(y>180)&(x>140) if t==2 else (x+y>180)
 elif window=='B3_A1_C4':z=(x<290)&(y<400)
 return z

def droplets(im,zone):
 circles=cv2.HoughCircles(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),cv2.HOUGH_GRADIENT,1,minDist=9,param1=65,param2=14,minRadius=3,maxRadius=24)
 water=np.zeros((448,448),np.uint8);accepted=[]
 if circles is not None:
  for x,y,r in circles[0]:
   ix,iy=int(round(x)),int(round(y))
   if not(0<=ix<448 and 0<=iy<448 and zone[iy,ix]):continue
   cv2.circle(water,(ix,iy),max(1,int(round(r-.8))),1,-1);accepted.append([float(x),float(y),float(r)])
 return water.astype(bool),accepted

allrecords=[];audit=[];relationships=[];panels={}
for row in rows:
 id=row['image_id'];window=row['window_id'];t=int(row['period']);im=np.array(Image.open(KIT/row['image_path']).convert('RGB'));water,circles=droplets(im,wet_zone(window,t))
 scene=np.zeros((448,448),np.uint8);scene[water]=2
 mask=np.array(Image.open(OLD/'candidate_masks'/(id+'.png')));attrs={k:np.array(Image.open(OLD/'attributes'/(id+'_'+k+'.png'))) for k in ['color','temporal','viability']};instance=np.array(Image.open(OLD/'attributes'/(id+'_trace_ids.png'))).astype(np.uint8);cohort=np.zeros_like(mask)
 if window=='B3_A0_C0':
  mask[:]=64;instance[:]=0
  for x in attrs.values():x[:]=0
  for r in reference['traces']:
   own=np.zeros((448,448),bool)
   for seg in r['periods'].get(str(t),[]):own|=line_mask(seg,r['width'])
   if not own.any():continue
   mask[own]=255;instance[own]=r.get('root_id',r['id']);origin=r['origin'] if t==1 or r['correspondence_status']=='visually_supported' else 0;cohort[own]=origin
   if t==1:cohort[own]=1;attrs['temporal'][own]=0
   elif r.get('state_by_period',{}).get(str(t))==1:attrs['temporal'][own]=1
   else:
    earlier=np.zeros_like(own)
    for seg in r['periods'].get(str(t-1),[]):earlier|=line_mask(seg,r['width'])
    if earlier.any() and r['correspondence_status']=='visually_supported':attrs['temporal'][own&(distance_transform_edt(~earlier)<=6)]=3
   relationships.append(dict(image_id=id,local_root_id=r['id'],first_observed_cohort=int(origin),visible=True,correspondence_status=r['correspondence_status'],source='period_specific_visual_trace_guided_by_user_examples'))
  for ext in reference['extensions']:
   if ext['period']!=t:continue
   own=line_mask(ext['points'],ext['width']);mask[own]=255;cohort[own]=1;instance[own]=ext['root_id'];attrs['temporal'][own]=2
  for u in reference['uncertain']:
   if u['period']==t:
    un=line_mask(u['points'],1.1)&(mask!=255);mask[un]=128
  # Only explicitly selected soil interiors are filled, not every unlabeled pixel.
  soilpoly={1:[(90,90),(170,85),(205,128),(275,160),(330,210),(360,265),(340,294),(265,320),(180,287),(80,278),(20,268),(25,215),(45,165),(65,120)],2:[(0,116),(80,109),(154,81),(195,107),(225,131),(285,150),(348,170),(390,235),(447,280),(447,430),(0,430)],3:[(0,118),(70,111),(150,90),(190,110),(240,142),(290,151),(348,174),(390,233),(447,275),(447,419),(0,419)],4:[(0,110),(80,108),(150,88),(190,110),(237,140),(289,154),(345,176),(388,231),(447,274),(447,419),(0,419)]}[t]
  soil=Image.new('L',(448,448));ImageDraw.Draw(soil).polygon(soilpoly,fill=1);scene[(np.array(soil)>0)&~water]=3
  source='assistant_period_specific_visual_trace_user_examples_no_ridge_snapping';status='example_retraced_pending_person_review'
 else:
  # Retract short isolated pieces and portions tangent to detected water rims. No new roots are invented.
  root=mask==255;sk=skeletonize(root);long=remove_small_objects(sk,min_size=9,connectivity=2);retract=root&(cv2.dilate(long.astype('uint8'),np.ones((3,3),np.uint8))==0)
  yy,xx=np.indices(root.shape);mass=cv2.GaussianBlur(root.astype('float32'),(0,0),2)+1e-5
  ex=cv2.GaussianBlur((xx*root).astype('float32'),(0,0),2)/mass;ey=cv2.GaussianBlur((yy*root).astype('float32'),(0,0),2)/mass
  cxx=cv2.GaussianBlur((xx*xx*root).astype('float32'),(0,0),2)/mass-ex*ex;cyy=cv2.GaussianBlur((yy*yy*root).astype('float32'),(0,0),2)/mass-ey*ey;cxy=cv2.GaussianBlur((xx*yy*root).astype('float32'),(0,0),2)/mass-ex*ey
  angle=.5*np.arctan2(2*cxy,cxx-cyy)
  for cx,cy,rad in circles:
   dx=xx-cx;dy=yy-cy;dist=np.sqrt(dx*dx+dy*dy);rim=np.abs(dist-rad)<1.5;tangent=np.abs((np.cos(angle)*dx+np.sin(angle)*dy)/np.maximum(dist,1))<.4;retract|=root&rim&tangent
  mask[retract]=128;instance[retract]=0
  for a in attrs.values():a[mask!=255]=0
  # Remove previous automatic claims of newborn/persistence: root identity needs actual review.
  attrs['temporal'][:]=0;attrs['viability'][:]=0;cohort[(mask==255)&(t==1)]=1
  source='prior_prefill_with_water_rim_and_fragment_retractions_temporal_claims_removed';status='needs_detailed_root_review'
 root=mask==255;scene[root]=1;scene[mask==128]=5
 # Root/non-root and object identity remain separate. Known soil/water becomes background only for segmentation.
 mask[np.isin(scene,[2,3,4])]=0
 for a in [*attrs.values(),cohort,instance]:a[~root]=0
 if t==1:attrs['temporal'][root]=0;cohort[root]=1
 for folder,a in [('masks',mask)]:Image.fromarray(a).save(OUT/folder/(id+'.png'))
 for name,a in {**attrs,'cohort':cohort,'scene':scene,'instance':instance}.items():Image.fromarray(a).save(OUT/'attributes'/(id+'_'+name+'.png'))
 colors=np.zeros_like(im);colors[root]=(37,99,235)
 for value,color in [(1,(22,163,74)),(2,(234,179,8)),(3,(147,51,234))]:colors[root&(attrs['temporal']==value)]=color
 ov=im.copy();ov[root]=(im[root]*.2+colors[root]*.8).astype('uint8');Image.fromarray(ov).save(OUT/'overlays'/'roots'/(id+'.png'));Image.fromarray(np.dstack([colors,np.where(root,255,0).astype('uint8')])).save(OUT/'roots_only'/(id+'.png'))
 clscolors={1:(37,99,235),2:(6,182,212),3:(174,135,89),4:(180,180,180),5:(249,115,22)};so=im.copy()
 for value,color in clscolors.items():sel=scene==value;so[sel]=(im[sel]*.45+np.array(color)*.55).astype('uint8')
 Image.fromarray(so).save(OUT/'overlays'/'scene'/(id+'.png'));(OUT/'originals'/(id+'.png')).write_bytes((KIT/row['image_path']).read_bytes())
 def runs(a):
  b=a.ravel();change=np.r_[True,b[1:]!=b[:-1]];idx=np.flatnonzero(change);values=b[idx];counts=np.diff(np.r_[idx,len(b)]);return np.column_stack([values,counts]).ravel().tolist()
 record=dict(name=id+'.png',image_sha256=row['image_sha256'],reviewed=False,evidence='',annotator='',prefill_origin=source,review_status=status,**{k:runs(a) for k,a in dict(mask=mask,**attrs,cohort=cohort,scene=scene,instance=instance).items()});allrecords.append(record)
 meta=dict(image_id=id,status=status,label_origin=source,reviewed_entire_image=False,original_byte_exact=True,root_first_observed_and_current_state_are_independent=True,unseen_roots_not_inferred=True,object_scene_encoding={'unreviewed':0,'root':1,'water_droplet':2,'soil':3,'other_background':4,'uncertain':5},source_cohort_encoding={'unknown':0,'T1_existing':1,'first_observed_T2':2,'first_observed_T3':3,'first_observed_T4':4},droplet_regions='circle proposals restricted to visually wet zones; require review',no_birth_or_death_proven=True)
 (OUT/'metadata'/(id+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');audit.append(dict(image_id=id,root_pixels=int(root.sum()),baseline_root_pixels=int(((cohort==1)&root).sum()),root_state_unclassified_pixels=int((root&(attrs['temporal']==0)).sum()),water_pixels=int((scene==2).sum()),soil_pixels=int((scene==3).sum()),unreviewed_pixels=int((scene==0).sum()),review_status=status))
 if window not in panels:panels[window]=Image.new('RGB',(960,1000),'white')
 panel=panels[window];ox=16+480*((t-1)%2);oy=35+500*((t-1)//2);ImageDraw.Draw(panel).text((ox,oy-25),id,fill='black',font=font);panel.paste(Image.fromarray(ov),(ox,oy))
for window,p in panels.items():p.save(OUT/'review_panels'/(window+'.png'))
(OUT/'review_project.json').write_text(json.dumps(dict(kind='B3_reviewable_annotations',version=2,encoding='rle_value_count',not_training_ground_truth=True,records=allrecords),separators=(',',':')))
(OUT/'annotation_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'root_relationships.json').write_text(json.dumps(relationships,indent=2)+'\n');print('REVISION_READY',len(allrecords),'T1_IS_BASELINE_ONLY')
