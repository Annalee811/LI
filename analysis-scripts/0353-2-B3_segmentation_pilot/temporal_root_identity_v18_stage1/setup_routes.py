import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent
for m in ['unet','deeplab','segformer']:
 for name in ['learn_routes','cache']:
  cmd=[sys.executable,str(here/(name+'.py')),'--run',str(a.run),'--roots',str(a.roots),'--model',m]
  if name=='cache':cmd+=['--previous',str(a.previous)]
  subprocess.run(cmd,check=True)
