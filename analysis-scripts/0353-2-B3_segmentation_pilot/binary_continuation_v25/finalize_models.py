"""Resume final evaluation model-by-model, reusing preserved outputs where eligible."""
import argparse,os,shutil,subprocess,sys,time
from pathlib import Path
import common as c

def copy_same(src,dst):
    if src.is_dir():
        dst.mkdir(parents=True,exist_ok=True)
        for f in src.iterdir():copy_same(f,dst/f.name)
    else:
        if dst.exists():assert c.sha(src)==c.sha(dst),'Existing result differs: '+str(dst)
        else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent;pending=list(c.MODELS)
    while pending:
        for m in pending[:]:
            out=a.run/m;training=out/'training_run.json'
            if not training.exists():continue
            t=c.read(training);old=a.source/m
            if (out/'prediction_complete.json').exists():pending.remove(m);continue
            selection=c.read(old/'selection.json');selection.update(selected_epoch=t['best_epoch'],checkpoint_SHA256=t['checkpoint_SHA256'],threshold=t['threshold'],baseline_retained=t['baseline_retained'],test_seen=False)
            c.js(out/'selection.json',selection);config=c.read(old/'config.json');config.update(continuation=c.read(a.run/'preparation.json'),epochs_completed=t['epochs_completed'],best_epoch=t['best_epoch']);c.js(out/'config.json',config)
            if t['baseline_retained']:
                ck='best_weights_fp16.npz' if m=='deeplab' else 'best.pt';assert c.sha(out/ck)==c.sha(old/ck)
                for name in ['probabilities','upper','metrics.json','val_per_tile_metrics.csv','test_per_tile_metrics.csv']:copy_same(old/name,out/name)
                c.js(out/'prediction_complete.json',dict(state='complete',original_final_outputs_reused_byte_for_byte=True,checkpoint_SHA256=c.sha(out/ck)));print('PRESERVED_ORIGINAL_FINAL_OUTPUTS',m,flush=True)
            elif m in ['unet','segformer']:
                with (a.run/(m+'_final_prediction.log')).open('a') as log:subprocess.run([sys.executable,str(here/'finish_torch.py'),'--run',str(a.run),'--model',m],stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                cmd=['docker','--host=unix:///var/run/docker.sock','run','--rm','--name','b3-root-v25-predict','--network','none','--tmpfs','/tmp:rw,size=128m','--tmpfs','/ram-python:rw,exec,size=1300m','--mount','type=bind,src='+str(Path.cwd())+',dst=/work','--workdir','/work','-e','OMP_NUM_THREADS=4','-e','PYTHONDONTWRITEBYTECODE=1','b3-tf112-mkl:local','sh','analysis-scripts/0353-2-B3_segmentation_pilot/binary_continuation_v25/container_entry.sh','--run',str(a.run),'--source',str(a.source),'--vendor','results/B3_uploaded_deeplab_v1/vendor','--threads','4','--predict-only']
                env=os.environ.copy()
                for key in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:env.pop(key,None)
                with (a.run/'deeplab_final_prediction.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
                c.js(out/'prediction_complete.json',dict(state='complete',original_final_outputs_reused_byte_for_byte=False,checkpoint_SHA256=t['checkpoint_SHA256']))
            pending.remove(m);c.js(a.run/'finalization_status.json',dict(completed=[m for m in c.MODELS if m not in pending],pending=pending));print('FINALIZED',m,flush=True)
        if pending:time.sleep(15)
    with (a.run/'figure_build.log').open('a') as log:subprocess.run([sys.executable,str(here/'figures.py'),'--run',str(a.run),'--source',str(a.source)],stdout=log,stderr=subprocess.STDOUT,check=True)
    c.js(a.run/'completion.json',dict(state='complete',models=c.MODELS,figures='figures',historical_loss_origin='Measured replay, not originally recorded validation loss',old_results_preserved=True));print('ALL_MODELS_AND_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':main()
