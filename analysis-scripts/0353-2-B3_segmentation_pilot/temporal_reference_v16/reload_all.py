"""Independently reload every selected head without reading reference labels."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent
for m in ['unet','deeplab','segformer']:
 subprocess.run([sys.executable,str(here/'predict_RGB.py'),'--head',str(a.run/m/'best_temporal_head.pt'),'--decoder',str(a.run/m/'decoder_selection.json'),'--rgb-dir',str(a.run/'source_RGB/0353-2-B3'),'--root-dir',str(a.roots/m/'upper'),'--output',str(a.run/'label_free_reload'/m),'--verify-against',str(a.run/m/'four_period_prediction')],check=True)
