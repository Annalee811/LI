"""Detached resumable sequence+graph pipeline, initial and optimized outputs preserved."""
import fcntl,os,subprocess,sys
from pathlib import Path
from prepare import OUT,MODELS,js
def main():
    lock=(OUT/'orchestrator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker present')
    here=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(OUT/'mpl-cache'))
    def stage(script,args=(),name=None):
        with (OUT/(name or script.replace('.py','.log'))).open('a',buffering=1) as f:
            child=subprocess.Popen([sys.executable,'-B',str(here/script),*args],stdout=f,stderr=subprocess.STDOUT,env=env)
            js(OUT/'pipeline_status.json',dict(stage=script,args=list(args),child_pid=child.pid,orchestrator_pid=os.getpid()))
            if child.wait():raise RuntimeError('Stage failed; inspect '+str(f.name))
    if not (OUT/'preflight_verification.json').exists():stage('verify.py')
    if not (OUT/'baseline_preserved_figures_complete.json').exists():stage('baseline_figures.py')
    for m in MODELS:
        for variant in ['convlstm_gat','control']:
            if not (OUT/(m+'_'+variant)/'completion.json').exists():stage('train.py',['--model',m,'--variant',variant],m+'_'+variant+'_training.log')
    if not (OUT/'evaluation_verification.json').exists():stage('evaluate.py')
    stage('figures.py')
    js(OUT/'completion.json',dict(state='complete',epochs_each=100,modules=6,latest_source='20',baseline_original_preserved=True,ConvLSTM_and_GAT_trained=True,binary_geometry_unchanged=True))
    js(OUT/'pipeline_status.json',dict(stage='complete'))
if __name__=='__main__':
    try:main()
    except Exception as e:js(OUT/'interruption.json',dict(error=str(e),recovery='Fix logged failure and resume same source/run_all.py; completed modules are preserved'));raise
