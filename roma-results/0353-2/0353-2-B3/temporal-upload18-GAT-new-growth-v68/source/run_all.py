"""Detached CPU pipeline. Preserve base training and checkpoint every new epoch."""
import fcntl,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from prepare import OUT,BASE,MODELS,build_geometry,add_baseline,js

def main():
    OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    lock=(OUT/'orchestrator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker present')
    here=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(OUT/'mpl-cache'))
    def stage(script,args=(),name=None):
        with (OUT/(name or script.replace('.py','.log'))).open('a',buffering=1) as log:
            p=subprocess.Popen([sys.executable,'-B',str(here/script),*args],stdout=log,stderr=subprocess.STDOUT,env=env)
            js(OUT/'pipeline_status.json',dict(stage=script,args=list(args),child_pid=p.pid,orchestrator_pid=os.getpid()))
            code=p.wait()
            if code:raise RuntimeError('Stage failed: '+script+'; inspect '+str(log.name))
    for m in MODELS:
        if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped')
        build_geometry(m)
    checked=set()
    for m in MODELS:
        while not add_baseline(m):
            if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped')
            js(OUT/'pipeline_status.json',dict(stage='waiting_for_current_baseline_full_inference',model=m,base=str(BASE),orchestrator_pid=os.getpid()))
            time.sleep(20)
        # Verify each newly available full graph before its gradients are used.
        stage('verify.py',['--model',m],name='preflight.log')
        for kind in ['gat','mlp']:
            if not (OUT/(m+'_'+kind)/'completion.json').exists():stage('train.py',['--model',m,'--kind',kind],m+'_'+kind+'_training.log')
    stage('evaluate.py');stage('figures.py')
    js(OUT/'completion.json',dict(state='complete',models=MODELS,epochs_each=100,new_modules=6,temporal_source='upload18',connection_source='upload17 user_defined IDs',binary_geometry_unchanged=True))
    js(OUT/'pipeline_status.json',dict(stage='complete',orchestrator_pid=os.getpid()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        js(OUT/'interruption.json',dict(error=str(e),recovery='Fix the actual logged failure and resume this same source/run_all.py; completed modules are not retrained'))
        raise
