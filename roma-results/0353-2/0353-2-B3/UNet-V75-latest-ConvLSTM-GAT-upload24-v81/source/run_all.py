"""Only U-Net local continuation; preserve designated base, other models and source24."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
from prepare import OUT,HERE,js

def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'orchestrator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/mpl-v80')
 def stage(name):
  with (OUT/(name+'.log')).open('a',buffering=1) as f:
   child=subprocess.Popen([sys.executable,'-B',str(HERE/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT,env=env);js(OUT/'pipeline_status.json',dict(stage=name,pid=child.pid,orchestrator_pid=os.getpid(),model='U-Net only',base='V75 AFTER selected45',learned_module='Latest V79 selected92',reference='24'))
   if child.wait():raise RuntimeError('Stage failed: '+str(OUT/(name+'.log')))
 if not (OUT/'preparation.json').exists():stage('prepare')
 if not (OUT/'preflight_verification.json').exists():stage('verify')
 if not (OUT/'completion.json').exists():stage('train')
 if not (OUT/'evaluation_verification.json').exists():stage('evaluate')
 if not (OUT/'figure_manifest.json').exists():stage('figures')
 if not (OUT/'publication_complete.json').exists():stage('publish')
 js(OUT/'pipeline_status.json',dict(stage='complete',model='U-Net only',base='V75 AFTER selected45',learned_module='Latest V79 selected92',reference='24'))
if __name__=='__main__':
 try:main()
 except Exception as e:js(OUT/'interruption.json',dict(error=str(e),recovery='Fix logged failure and resume same run_all.py; completed real epochs retained'));raise
