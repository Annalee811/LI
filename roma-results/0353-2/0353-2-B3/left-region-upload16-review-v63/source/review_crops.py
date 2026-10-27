import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path('/workspace/LI');O=R/'results/B3_left_region_review_v63_20261027';D=json.loads((O/'source_progress_16.json').read_text());S=D['root_structure_annotation'];H={h['root_id']:h for h in S['root_track_hierarchy']}
def panel(name,bbox,z=2):
 w,h=bbox[2]-bbox[0],bbox[3]-bbox[1];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);im=Image.new('RGB',(w*z*2,(h*z+30)*2),'white');draw=ImageDraw.Draw(im)
 for i,f in enumerate(S['periods']):
  rgb=Image.open(R/f"results/B3_temporal_unet_aligned_v21_20261021/data/upper/{f['period']}_RGB.png").convert('RGB').crop(bbox).resize((w*z,h*z));dr=ImageDraw.Draw(rgb)
  for s in f['segments']:
   pp=[p for p in s['points'] if bbox[0]<=p[0]<bbox[2] and bbox[1]<=p[1]<bbox[3]]
   if not pp:continue
   defined=H[s['root_id']].get('annotation_status_v60')!='unassigned';color='#00ff50' if defined else '#ff30dc'
   dr.line([((p[0]-bbox[0])*z,(p[1]-bbox[1])*z) for p in s['points']],fill=color,width=2)
   if s['length_px']>=8:
    p=pp[len(pp)//2];label=H[s['root_id']]['label'] if defined else str(s['index']);dr.text(((p[0]-bbox[0])*z,(p[1]-bbox[1])*z),label,font=font,fill='yellow',stroke_width=2,stroke_fill='black')
  x=(i%2)*w*z;y=(i//2)*(h*z+30);im.paste(rgb,(x,y+30));draw.text((x+5,y+5),f['period'],font=font,fill='black')
 im.save(O/(name+'.png'))
for name,bbox,z in [('upper_detail',(250,180,450,360),3),('middle_detail',(0,350,510,590),2),('lower_detail',(0,800,480,1108),2)]:panel(name,bbox,z)
if __name__=='__main__':
 print('Formal labels:',sorted(h['label'] for h in H.values() if h.get('annotation_status_v60')!='unassigned'))
