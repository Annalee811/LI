"""Replace only the illustrative local window of frozen V44 temporal outputs."""
from pathlib import Path
import hashlib,json,csv,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageCms
from scipy.ndimage import label as component_labels
R=Path('/workspace/LI');RUN=R/'results/B3_temporal_initialization_v43_20261024';SOURCE=R/'results/B3_temporal_paper_v44_20261024';O=R/'results/B3_temporal_local_window_v64_20261027'
FONT=R/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf';PERIODS=['T1','T2','T3','T4'];MODELS=['unet','deeplab','segformer'];NAMES=['Reference','U-Net','DeepLabV3+','SegFormer-B0']
PAL=np.full((256,3),148,np.uint8);PAL[:5]=[[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 sys.path.insert(0,str(R/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position('0353-2-B3')=='0353-2-B3'
 candidate=json.loads((O/'all_reference_window_candidates.json').read_text())[0];x,y,size=candidate['x'],candidate['y'],candidate['size'];assert (x,y,size)==(1440,120,384)
 roi=[x,y,x+size,y+size];sl=np.s_[y:y+size,x:x+size]
 sources=[RUN/f'data/upper/{t}_RGB.png' for t in PERIODS]+[RUN/f'data/current/{t}_classes.png' for t in PERIODS]+[SOURCE/f'{m}/full/{t}_classes.png' for m in MODELS for t in PERIODS]
 sources +=[SOURCE/'metrics_table.json',SOURCE/'Fig1_Four_period_temporal_root_comparison.tif']
 records=[dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for p in sources]
 rgbs={t:np.array(Image.open(RUN/f'data/upper/{t}_RGB.png').convert('RGB')) for t in PERIODS};ref={t:np.array(Image.open(RUN/f'data/current/{t}_classes.png')) for t in PERIODS};pred={(m,t):np.array(Image.open(SOURCE/f'{m}/full/{t}_classes.png')) for m in MODELS for t in PERIODS}
 image=Image.new('RGB',(6600,3660),'white');dr=ImageDraw.Draw(image);panels=[];texts=[];arrows=[];counts=[]
 def text(x,y,v,size=48,anchor='mt'):
  font=ImageFont.truetype(str(FONT),size);b=dr.textbbox((x,y),v,font=font,anchor=anchor);assert min(b[:2])>=0 and b[2]<=image.width and b[3]<=image.height,(v,b);dr.text((x,y),v,font=font,fill='black',anchor=anchor);texts.append(dict(text=v,font='Times New Roman Bold',color='black',box=b))
 def place(a,x,y,w,h,tag,categorical=False):
  assert a.shape[:2]==(size,size);image.paste(Image.fromarray(a).resize((w,h),Image.Resampling.NEAREST if categorical else Image.Resampling.LANCZOS),(x,y));dr.rectangle((x,y,x+w-1,y+h-1),outline='#d8dde3',width=2);panels.append(dict(tag=tag,box=[x,y,w,h],native_shape=list(a.shape),ROI_xyxy=roi,categorical_nearest_neighbor=categorical))
 def scale(x,y,w,h):
  n=50;length=round(w*n/size);px=x+18;py=y+h-68;dr.rectangle((px-7,py-6,px+length+10,py+52),fill='white');dr.rectangle((px,py+34,px+length,py+40),fill='black');text(px+length/2,py,'50 px',26)
 def overlay(rgb,cl):
  out=rgb.copy();mask=cl>0;out[mask]=np.rint(.25*rgb[mask]+.75*PAL[cl[mask]]).astype('uint8');assert np.array_equal(out[~mask],rgb[~mask]);return out
 def point(cl,k):
  lab,n=component_labels(cl==k,structure=np.ones((3,3)));assert n;areas=np.bincount(lab.ravel());areas[0]=0;chosen=areas.argmax();points=np.argwhere(lab==chosen);center=points.mean(0);yy,xx=points[((points-center)**2).sum(1).argmin()];return int(xx),int(yy),int(areas[chosen])
 def arrow(ax,ay,px,py,mark):
  # The arrows identify reference positions only; they never recolor model masks.
  tx=max(ax+35,min(ax+525,px+(90 if px<ax+280 else -90)));ty=max(ay+45,min(ay+515,py+(-65 if py>ay+100 else 65)));dr.line((tx,ty,px,py),fill='black',width=3);v=np.array([px-tx,py-ty],float);v/=max(1,np.linalg.norm(v));perp=np.array([-v[1],v[0]]);tip=np.array([px,py]);b=tip-v*15;dr.polygon([tuple(tip),tuple(b+perp*7),tuple(b-perp*7)],fill='black');font=ImageFont.truetype(str(FONT),46);bbox=dr.textbbox((tx,ty),mark,font=font,anchor='mm');dr.rectangle((bbox[0]-5,bbox[1]-4,bbox[2]+5,bbox[3]+4),fill='white');text(tx,ty,mark,46,anchor='mm')
 text(3300,22,'Temporal root classification: matched local comparison',86)
 text(3300,130,'0353-2-B3 | Same 384 × 384 px illustrative window across periods and methods',43)
 for i,t in enumerate(PERIODS):
  bx=90+(i%2)*3280;by=230+(i//2)*1570;start=bx+650;tile=560;yy=by+170;whitey=yy+585;rawy=by+560
  text(bx,by,t,78,anchor='lt');text(bx+195,by+90,'Original RGB',53)
  rgb=rgbs[t][sl];place(rgb,bx,rawy,390,390,t+'/local_RGB');scale(bx,rawy,390,390)
  text(bx+515,yy+280,'Overlay',36,anchor='mm');text(bx+515,whitey+280,'Classes',36,anchor='mm')
  for j,name in enumerate(NAMES):text(start+j*580+280,by+90,name,53)
  for j,m in enumerate(['reference']+MODELS):
   cl=(ref[t] if m=='reference' else pred[m,t])[sl];assert cl.shape==(384,384);assert set(np.unique(cl)).issubset({0,1,2,3,4,255})
   xx=start+j*580;place(overlay(rgb,cl),xx,yy,tile,tile,t+'/'+m+'/overlay');place(PAL[cl],xx,whitey,tile,tile,t+'/'+m+'/white_classes',True)
   counts.append(dict(period=t,method=m,counts={str(k):int((cl==k).sum()) for k in [0,1,2,3,4,255]}))
  if t!='T1':
   for k,mark in [(2,'N'),(3,'E')]:
    xx0,yy0,area=point(ref[t][sl],k);assert area>0;arrows.append(dict(period=t,class_code=k,mark=mark,point_native_local=[xx0,yy0],component_pixels=area))
    for rowy in [yy,whitey]:arrow(start,rowy,start+xx0*tile/size,rowy+yy0*tile/size,mark)
 items=[('T1 baseline',4),('Persistent',1),('Newly observed (N)',2),('Tip extension (E)',3)]
 for j,(name,k) in enumerate(items):
  xx=290+j*1570;dr.rectangle((xx,3400,xx+55,3455),fill=tuple(PAL[k]));text(xx+80,3400,name,49,anchor='lt')
 text(3300,3500,'N / E arrows mark reference locations; original masks and model predictions are unchanged.',42)
 text(3300,3572,'T1 shows baseline roots; T2–T4 classes are relative to the immediately preceding period.',42)
 assert len(panels)==36 and len(arrows)==6
 path=O/'Fig_Four_period_temporal_local_new_and_extension.tif';image.save(path,compression='tiff_adobe_deflate',tiffinfo={317:2},dpi=(600,600),icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes());prev=image.copy();prev.thumbnail((1800,1800));prev.save(O/'preview.png')
 with Image.open(path) as check:assert check.mode=='RGB' and check.size==(6600,3660) and tuple(round(v) for v in check.info['dpi'])==(600,600)
 # Record overlap with prior spatial partitions; no independent-test claim.
 overlaps=[]
 with (RUN/'data/spatial_manifest.csv').open(encoding='utf-8-sig') as fd:
  for r in csv.DictReader(fd):
   rx,ry,rw,rh=[int(r[k]) for k in ['x','y','width','height']];area=max(0,min(x+size,rx+rw)-max(x,rx))*max(0,min(y+size,ry+rh)-max(y,ry))
   if area:overlaps.append(dict(tile_id=r['tile_id'],split=r['split'],overlap_pixels=area))
 selection=dict(ROI_xyxy=roi,previous_ROI_xyxy=[1428,852,1684,1108],selected=candidate,selection='Reference-only log coverage of new and extension across T2–T4; minimum 60 pixels per class/date; excludes old ROI; no model outputs used',qualitative_illustrative_window=True,prior_spatial_split_overlaps=overlaps,not_claimed_as_independent_test_window=True,arrows=arrows)
 js(O/'ROI_selection.json',selection)
 assert all(sha(Path(r['path']))==r['SHA256'] for r in records)
 verification=dict(passed=True,source_version='V44 saved predictions from V43 100-epoch temporal heads',source_records=records,no_training=True,no_inference=True,reference_and_prediction_files_unchanged=True,metrics_unchanged=True,previous_figure_preserved=True,ROI_xyxy=roi,same_ROI_for_all_36_panels=True,each_T2_T3_T4_reference_has_new_and_extension=True,counts=counts,reference_arrows=arrows,all_labels_black_bold_Times_New_Roman=True,typography=texts,panels=panels,output=dict(file=path.name,bytes=path.stat().st_size,SHA256=sha(path),dimensions=[6600,3660],dpi=600,mode='RGB'))
 js(O/'verification.json',verification)
 with (O/'class_counts.csv').open('w',newline='') as fd:
  w=csv.writer(fd);w.writerow(['period','method','background','persistent','newly_observed','tip_extension','baseline','ignored']);[w.writerow([r['period'],r['method'],*[r['counts'][str(k)] for k in [0,1,2,3,4,255]]]) for r in counts]
 print(json.dumps(dict(passed=True,ROI=roi,output=verification['output'],reference_counts=[r for r in counts if r['method']=='reference']),indent=2))
if __name__=='__main__':main()
