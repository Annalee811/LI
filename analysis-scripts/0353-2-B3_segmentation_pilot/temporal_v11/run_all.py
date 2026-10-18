"""Sequential resumable three-backbone temporal-head experiment."""
import argparse,os,subprocess,sys
from pathlib import Path
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--samples',type=int,default=48);a=p.parse_args()
    here=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True)
    for model in c.MODELS:
        if (a.output/model/'training_run.json').exists():print('RESUME_COMPLETE_MODEL',model,flush=True);continue
        cmd=[sys.executable,str(here/'train.py'),'--data',str(a.data),'--roots',str(a.roots),'--output',str(a.output),'--model',model,'--epochs',str(a.epochs),'--samples',str(a.samples),'--previous',str(a.previous)]
        with (a.output/(model+'_training.log')).open('w') as log:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=os.environ.copy())
            for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if proc.wait():raise SystemExit(proc.returncode)
    print('ALL_THREE_TEMPORAL_MODELS_TRAINED',flush=True)
if __name__=='__main__':main()
