"""Sequential new U-Net training then strictly local ConvLSTM/GAT optimization."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
from prepare import OUT,ROOT
import common as c

def main():
 lock=(OUT/'orchestrator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);here=Path(__file__).resolve().parent;local=ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/unet_local_upload24_v79';env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/mpl-v79')
 if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped by user')
 def stage(script,args=(),label=None):
  name=label or script.stem;log=OUT/(name+'.log')
  with log.open('a',buffering=1) as f:
   child=subprocess.Popen([sys.executable,'-B',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT);c.js(OUT/'pipeline_status.json',dict(stage=name,pid=child.pid,orchestrator_pid=os.getpid(),reference='24'))
   if child.wait():raise RuntimeError('Stage failed: '+str(log))
 if not (OUT/'unet/completion.json').exists():stage(here/'train.py',['--run',str(OUT),'--model','unet','--threads','2'],'unet_training')
 if not (OUT/'full_inference_verification.json').exists():stage(here/'infer.py',label='unet_full_inference')
 dest=ROOT/'results/B3_UNet_local_upload24_v79_20261028'
 if not (dest/'preparation.json').exists():stage(local/'prepare.py',label='local_preparation')
 if not (dest/'preflight_verification.json').exists():stage(local/'verify.py',label='local_preflight')
 if not (dest/'completion.json').exists():stage(local/'train.py',label='local_training')
 if not (dest/'evaluation_verification.json').exists():stage(local/'evaluate.py',label='local_evaluation')
 stage(local/'figures.py',label='local_figures');stage(local/'publish.py',label='local_publish');c.js(OUT/'pipeline_status.json',dict(stage='complete',reference='24',result=str(dest)))
if __name__=='__main__':
 try:main()
 except Exception as e:c.js(OUT/'interruption.json',dict(error=str(e),recovery='Fix logged failure and resume source/run_all.py; completed real epochs retained'));raise
