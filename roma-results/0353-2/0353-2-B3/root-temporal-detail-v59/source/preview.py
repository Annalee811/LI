"""Native RGB review panels, not synthetic imagery."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_temporal_review_v59_20261026'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def main():
 d=json.loads((OUT/'root_structure_draft.json').read_text());st=d['root_structure_annotation'];audit=st['temporal_revision_v59'];hh={h['root_id']:h for h in st['root_track_hierarchy']};frames=st['periods'];rgb=[Image.open(ROOT/'results/B3_temporal_unet_aligned_v21_20261021/data/upper'/f'T{j+1}_RGB.png').convert('RGB') for j in range(4)]
 events=sorted([x for x in audit['birth_reviews'] if any(y['root_id']==x['root_id'] for y in audit['new_independent_numbering'])],key=lambda x:(x['first_seen_period'],-x['axis_length_px']))
 chosen=[]
 for t in ['T2','T3','T4']:chosen.extend([x for x in events if x['first_seen_period']==t][:4])
 sheet=Image.new('RGB',(1600,len(chosen)*225),'white');draw=ImageDraw.Draw(sheet)
 for row,e in enumerate(chosen):
  r=e['root_id'];first=int(e['first_seen_period'][1:])-1;ss=[s for s in frames[first]['segments'] if s['root_id']==r];p=[q for s in ss for q in s['points']];xs=[q[0] for q in p];ys=[q[1] for q in p];cx=(min(xs)+max(xs))/2;cy=(min(ys)+max(ys))/2;ww=max(300,max(xs)-min(xs)+60);height=max(160,max(ys)-min(ys)+40);x0=max(0,min(2541-ww,int(cx-ww/2)));y0=max(0,min(1108-height,int(cy-height/2)));box=(int(x0),int(y0),int(x0+ww),int(y0+height))
  for fi in range(4):
   im=rgb[fi].copy();dr=ImageDraw.Draw(im)
   for s in frames[fi]['segments']:
    if s['root_id']==r:dr.line(s['points'],fill='#00ffd5',width=3)
   im=im.crop(box);im.thumbnail((390,190));sheet.paste(im,(fi*400,row*225+30));draw.text((fi*400+3,row*225+4),f'{hh[r]["label"]} | T{fi+1} | first {e["first_seen_period"]}',font=FONT,fill='black')
 sheet.save(OUT/'new_root_four_period_review.png');sheet.save(OUT/'New_root_four_period_review.tif',compression='tiff_lzw',dpi=(300,300))
 # Compact panels for every applied temporal/identity correction.
 changes=audit['continuation_recoveries']+audit['temporal_pixel_corrections']
 sheet=Image.new('RGB',(1600,max(1,len(changes))*210),'white');draw=ImageDraw.Draw(sheet)
 for row,e in enumerate(changes):
  x,y=e['point'];box=(max(0,min(2241,x-150)),max(0,min(928,y-90)));box=(*box,box[0]+300,box[1]+180)
  for fi in range(4):
   im=rgb[fi].copy();dr=ImageDraw.Draw(im)
   for s in frames[fi]['segments']:
    if s['root_id']==e['root_id']:dr.line(s['points'],fill='#00ffd5',width=2)
   crop=im.crop(box);sheet.paste(crop,(fi*400,row*210+27));draw.text((fi*400+3,row*210+3),f'{hh[e["root_id"]]["label"]} | T{fi+1}',font=FONT,fill='black')
 sheet.save(OUT/'applied_corrections_review.png')
 # Same native T1 RGB, before/after the screenshot-directed identity repair.
 source=json.loads((OUT/'source_progress_14.json').read_text());repair=audit['T1_changes_explicitly_requested_in_screenshot'];oldh={h['root_id']:h for h in source['root_structure_annotation']['root_track_hierarchy']};bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',27);sheet=Image.new('RGB',(2541,2296),'white');draw=ImageDraw.Draw(sheet)
 for row,(obj,maph) in enumerate([(source,oldh),(d,hh)]):
  im=rgb[0].copy();dr=ImageDraw.Draw(im);seen=set();placed=[];parts=obj['root_structure_annotation']['periods'][0]['segments'];familyset={repair['upper_root_id']} if row==0 else {repair['upper_root_id'],repair['lower_root_id']}
  for s in sorted(parts,key=lambda s:-s['length_px']):
   h=maph[s['root_id']];family=h['family_root_id']
   if family not in familyset:continue
   color='#00df87' if family==repair['upper_root_id'] else '#21b9ff';dr.line(s['points'],fill=color,width=4)
   if s['root_id'] not in seen and s['length_px']>30:
    pt=s['points'][len(s['points'])//2];text=h['label'];width=dr.textlength(text,font=bold)+10;choices=[]
    for dy in [-35,12,-75,48,-115,84]:
     for dx in [8,40,-80,100,-150,160]:
      x=max(3,min(im.width-width-3,pt[0]+dx));y=max(3,min(im.height-39,pt[1]+dy));b=(x,y,x+width,y+36);overlap=any(not(b[2]+5<q[0] or q[2]+5<b[0] or b[3]+5<q[1] or q[3]+5<b[1]) for q in placed)
      if not overlap:choices.append((dx*dx+dy*dy,b))
    if choices:
     box=min(choices,key=lambda q:q[0])[1];dr.line([tuple(pt),(max(box[0],min(pt[0],box[2])),max(box[1],min(pt[1],box[3])))],fill=color,width=1);dr.rectangle(box,fill='white',outline=color,width=2);dr.text((box[0]+5,box[1]+1),text,font=bold,fill='black');placed.append(box)
    seen.add(s['root_id'])
  sheet.paste(im,(0,row*1148+40));title='Before: remote upper and lower axes share upload14 family0030' if row==0 else 'After: upper family0030 and existing lower family'+repair['lower_label']+' are separate';draw.text((12,row*1148+5),title,font=bold,fill='black')
 sheet.save(OUT/'T1_distant_family_before_after.tif',compression='tiff_adobe_deflate',dpi=(300,300));sheet.save(OUT/'T1_distant_family_before_after.png')
if __name__=='__main__':main()
