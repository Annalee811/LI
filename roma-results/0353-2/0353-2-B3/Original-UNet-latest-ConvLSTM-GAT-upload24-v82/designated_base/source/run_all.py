"""Detached restartable pipeline; no prior experiment checkpoint is loaded."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path

def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve()
    lock=(a.run/'orchestrator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.run/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker present')
    if (a.run/'completion.json').exists():print('ALREADY_COMPLETE',flush=True);return
    here=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(a.run/'matplotlib-cache'))
    state=dict(orchestrator_pid=os.getpid(),stage='features',GAT=False,reference='upload18',models={})
    write(a.run/'launch_state.json',state)
    def stage(script,log):
        with (a.run/log).open('a',buffering=1) as f:
            subprocess.run([sys.executable,'-B',str(here/script),'--run',str(a.run)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (a.run/'feature_verification.json').exists():stage('build_features.py','feature_build.log')
    if not (a.run/'preflight_verification.json').exists():stage('verify.py','preflight.log')
    jobs={};handles=[]
    for m in ['unet','deeplab','segformer']:
        if (a.run/m/'completion.json').exists():state['models'][m]=dict(state='complete');continue
        cmd=[sys.executable,'-B',str(here/'train.py'),'--run',str(a.run),'--model',m,'--threads',str(2 if m=='unet' else 1)]
        log=(a.run/(m+'_training.log')).open('a',buffering=1);handles.append(log)
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[m]=child;state['models'][m]=dict(state='running',pid=child.pid,started_unix=time.time(),command=cmd)
    state['stage']='training';write(a.run/'launch_state.json',state);write(a.run/'pipeline_status.json',state)
    last_plot=time.monotonic()
    while jobs:
        for m,child in list(jobs.items()):
            code=child.poll()
            if code is not None:
                state['models'][m].update(state='complete' if code==0 and (a.run/m/'completion.json').exists() else 'failed',exit_code=code)
                del jobs[m]
        write(a.run/'launch_state.json',state)
        if time.monotonic()-last_plot>900 or not jobs:
            stage('plot.py','plot.log');last_plot=time.monotonic()
        if jobs:time.sleep(10)
    for f in handles:f.close()
    if not all((a.run/m/'completion.json').exists() for m in state['models']):
        state['stage']='interrupted';write(a.run/'pipeline_status.json',state);raise SystemExit('Review actual branch logs; recover only this run')
    state['stage']='full_inference_and_class_figures';write(a.run/'pipeline_status.json',state)
    stage('finish.py','finish.log')
    state['stage']='complete';write(a.run/'pipeline_status.json',state)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
