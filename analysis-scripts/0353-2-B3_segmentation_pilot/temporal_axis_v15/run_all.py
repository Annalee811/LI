import argparse,subprocess,sys
from pathlib import Path
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--roots',required=True);p.add_argument('--previous',required=True);a=p.parse_args()
for m in c.MODELS:subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--run',a.run,'--roots',a.roots,'--previous',a.previous,'--model',m],check=True)
print('ALL_THREE_MANUAL_FIRST_HEADS_TRAINED',flush=True)
