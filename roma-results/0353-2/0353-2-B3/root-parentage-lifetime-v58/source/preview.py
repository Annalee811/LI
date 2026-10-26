"""Review figure made only from native RGB and actual identity traces."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_parentage_v58_20261026'
def main():
 d=json.loads((OUT/'root_structure_draft.json').read_text());st=d['root_structure_annotation'];hh={a['root_id']:a for a in st['root_track_hierarchy']};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);can=Image.new('RGB',(2542,1190),'white');dr=ImageDraw.Draw(can)
 for fi,f in enumerate(st['periods']):
  im=Image.open(ROOT/'results/B3_temporal_unet_aligned_v21_20261021/data/upper'/f"{f['period']}_RGB.png").convert('RGB');p=ImageDraw.Draw(im);seen=set()
  for s in sorted(f['segments'],key=lambda s:-s['length_px']):
   label=hh[s['root_id']]['label']
   if not(label=='0001' or label.startswith('0001-')):continue
   p.line(s['points'],fill='#00e5b0',width=3)
   if label not in seen and len(s['points'])>15:p.text(s['points'][len(s['points'])//2],label,font=font,fill='white',stroke_width=2,stroke_fill='black');seen.add(label)
  x=fi%2*1271;y=fi//2*595;can.paste(im.resize((1270,554)),(x,y+28));dr.text((x+5,y+4),f['period']+' | Family 0001 - identity proposals for review',font=font,fill='black')
 can.save(OUT/'family0001_continuity_review.png');can.save(OUT/'Family0001_four_period_continuity_review.tif',compression='tiff_lzw',dpi=(300,300))
if __name__=='__main__':main()
