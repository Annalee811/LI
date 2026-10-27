from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
R=Path('/workspace/LI');O=R/'results/B3_left_region_review_v63_20261027';D=json.loads((O/'review_project.json').read_text());S=D['root_structure_annotation'];P=S['regional_review_v63']['proposals'];H={h['root_id']:h for h in S['root_track_hierarchy']}
COLORS=['#e69f00','#009e73','#d55e00','#56b4e9','#cc79a7','#f0e442','#0072b2','#ff7f50','#a6d854','#e78ac3','#b3a200','#00c4bf'];OFFSETS={'0004-1':(380,245),'0007-7':(225,412),'0031':(355,535),'0032':(180,557),'0033':(5,610),'0034':(10,465),'0035':(230,855),'0036':(480,995),'0037':(0,1055),'0038':(160,1080),'0039':(280,1005),'0040':(445,1060)}
z=1.5;box=(0,150,750,1108);w=int(750*z);h=int((1108-150)*z);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',27)
page=Image.new('RGB',(w*2+60,(h+55)*2+110),'white');draw=ImageDraw.Draw(page);draw.text((20,12),'Left region: four-period root identity proposals',font=title,fill='black')
for j,f in enumerate(S['periods']):
 crop=Image.open(R/f"results/B3_temporal_unet_aligned_v21_20261021/data/upper/{f['period']}_RGB.png").convert('RGB').crop(box).resize((w,h));d=ImageDraw.Draw(crop)
 for seg in f['segments']:
  if H[seg['root_id']].get('annotation_status_v60') not in ['user_defined','reviewed_visible_axis']:continue
  d.line([(int(p[0]*z),int((p[1]-150)*z)) for p in seg['points']],fill='#76d7a0',width=1)
 for i,p in enumerate(P):
  ss=[v for v in f['segments'] if v['root_id']==p['root_id']];lx,ly=OFFSETS[p['label']];lx*=z;ly=(ly-150)*z;label=p['label'] if ss else p['label']+': unresolved';color=COLORS[i];bw=d.textbbox((0,0),label,font=font);tw=bw[2]-bw[0]+10;th=bw[3]-bw[1]+10;lx=min(w-tw-3,max(3,lx));ly=min(h-th-3,max(3,ly))
  for seg in ss:d.line([(int(q[0]*z),int((q[1]-150)*z)) for q in seg['points']],fill=color,width=3)
  if ss:
   longest=max(ss,key=lambda x:x['length_px']);p0=longest['points'][len(longest['points'])//2];d.line([(int(p0[0]*z),int((p0[1]-150)*z)),(int(lx),int(ly+th/2))],fill=color,width=1)
  d.rectangle((lx,ly,lx+tw,ly+th),fill='white',outline=color,width=3);d.text((lx+5,ly-bw[1]+5),label,font=font,fill='black')
 x=20+(j%2)*(w+20);y=65+(j//2)*(h+55);draw.text((x,y),f['period'],font=title,fill='black');page.paste(crop,(x,y+35))
draw.text((20,page.height-36),'Provisional identities; original masks unchanged. 0033 in T4 is unresolved. Human review required.',font=small,fill='black');page.save(O/'Four_period_left_region_review.tif',compression='tiff_lzw',dpi=(300,300));page.thumbnail((1200,1700));page.save(O/'review_preview.png');print('saved TIFF')
