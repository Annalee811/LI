"""Reproducible sequential three-head training, inference, and evaluation."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent
for model in ['unet','deeplab','segformer']:
 for name in ['train','predict']:
  args=[sys.executable,str(here/(name+'.py')),'--run',str(a.run),'--roots',str(a.roots),'--model',model]
  if name=='train':args+=['--previous',str(a.previous),'--epochs','32']
  with (a.run/(model+'_'+name+'.log')).open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('COMPLETED',model,name,flush=True)
subprocess.run([sys.executable,str(here/'report.py'),'--run',str(a.run)],check=True)
