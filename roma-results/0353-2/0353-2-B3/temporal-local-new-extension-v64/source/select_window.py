from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,json,math
R=Path('/workspace/LI');RUN=R/'results/B3_temporal_initialization_v43_20261024';O=R/'results/B3_temporal_local_window_v64_20261027';ts=['T2','T3','T4'];refs={t:np.array(Image.open(RUN/f'data/current/{t}_classes.png')) for t in ['T1']+ts};ii={(t,k):np.pad((refs[t]==k).astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0))) for t in ts for k in [1,2,3]}
def count(t,k,x,y,w):
 a=ii[t,k];return int(a[y+w,x+w]-a[y,x+w]-a[y+w,x]+a[y,x])
rows=[]
for w in [256,320,384]:
 for y in range(0,1108-w+1,24):
  for x in range(0,2541-w+1,24):
   if x<1684 and x+w>1428 and y<1108 and y+w>852:continue
   counts={t:{str(k):count(t,k,x,y,w) for k in [1,2,3]} for t in ts};v=[counts[t][str(k)] for t in ts for k in [2,3]]
   if min(v)<60:continue
   score=sum(math.log1p(c) for c in v)-2*math.log(w/256)
   rows.append(dict(x=x,y=y,size=w,counts=counts,score=score,minimum_new_or_extension_pixels=min(v)))
rows.sort(key=lambda r:r['score'],reverse=True);(O/'all_reference_window_candidates.json').write_text(json.dumps(rows[:100],indent=2));print('valid regions',len(rows));print(json.dumps(rows[:8],indent=2))
pal=np.full((256,3),148,np.uint8);pal[:5]=[[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]];font=ImageFont.truetype(str(R/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'),16)
chosen=[]
for r in rows:
 if all(abs(r['x']-q['x'])>100 or abs(r['y']-q['y'])>100 for q in chosen):chosen.append(r)
 if len(chosen)>=4:break
page=Image.new('RGB',(4*300,len(chosen)*350),'white');dr=ImageDraw.Draw(page)
for i,r in enumerate(chosen):
 for j,t in enumerate(['T1']+ts):
  x,y,w=r['x'],r['y'],r['size'];rgb=np.array(Image.open(RUN/f'data/upper/{t}_RGB.png').convert('RGB').crop((x,y,x+w,y+w)));cl=refs[t][y:y+w,x:x+w];mask=cl>0;rgb[mask]=np.rint(.25*rgb[mask]+.75*pal[cl[mask]]).astype('uint8');ox=j*300;oy=i*350;page.paste(Image.fromarray(rgb).resize((280,280)),(ox,oy+45));dr.text((ox+5,oy),f'{i+1} {t}: {x},{y} ({w})',font=font,fill='black');dr.text((ox+5,oy+23),f'New {int((cl==2).sum())} / Tip {int((cl==3).sum())}',font=font,fill='black')
page.save(O/'new_window_candidates.png');(O/'distinct_candidates.json').write_text(json.dumps(chosen,indent=2))
