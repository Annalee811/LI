"""Resumable teacher-reference temporal head fine-tuning; original audit separate."""
import argparse,subprocess,sys,time
from pathlib import Path
import common as c

def main():
    p=argparse.ArgumentParser()
    for key in ['run','roots','previous','progress']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent
    stages=[('prepare',None)]+[(s,m) for m in c.MODELS for s in ['train','predict']]+[('report',None),('build_review',None),('audit',None)]
    for stage,m in stages:
        label=(m+'_' if m else '')+stage
        cmd=[sys.executable,str(here/(stage+'.py')),'--run',str(a.run)]
        if stage not in ['build_review']:cmd+=['--roots',str(a.roots)]
        if m:cmd+=['--model',m]
        if stage in ['prepare','train']:cmd+=['--previous',str(a.previous)]
        if stage=='prepare':cmd+=['--progress',str(a.progress)]
        if stage=='train':cmd+=['--epochs',str(a.epochs)]
        c.js(a.run/'status.json',dict(stage=label,state='running'));print('START',label,flush=True)
        with (a.run/(label+'.log')).open('a') as log:
            log.write('\nSTAGE_START '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n');log.flush();r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            c.js(a.run/'status.json',dict(stage=label,state='failed',returncode=r.returncode,log=str(a.run/(label+'.log'))));raise SystemExit(r.returncode)
        c.js(a.run/'status.json',dict(stage=label,state='complete'));print('COMPLETED',label,flush=True)
    c.js(a.run/'completion.json',dict(state='complete',models=c.MODELS,evaluation='Teacher consistency and original annotation audit separately; external independent images 0'));c.js(a.run/'status.json',dict(stage='all',state='complete'));print('ALL_THREE_COMPLETE',flush=True)

if __name__=='__main__':main()
