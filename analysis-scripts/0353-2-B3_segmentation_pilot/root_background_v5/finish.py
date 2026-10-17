#!/usr/bin/env python3
"""After all fits: freeze validation selection, score, visualize, and package.

No temporal trainer or other-location prediction is invoked, regardless of scores.
Docker runs only the original uploaded DeepLab implementation.
"""
import argparse,os,subprocess,sys
from pathlib import Path
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=Path(__file__).resolve().parent;repo=Path.cwd().resolve()
    assert src.is_relative_to(repo),'Run from the workspace containing source, data and weights'
    for name in ['unet','segformer','deeplab','unet_precision_candidate']:
        assert (a.run/name/'training_run.json').exists(),'Finish all actual fitting before evaluation'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME='/tmp/B3_v5_cache',MPLCONFIGDIR='/tmp/B3_v4_mpl')
    def run(cmd,label):
        print('ROOT_FINALIZATION',label,flush=True)
        with (a.run/(label+'.log')).open('w') as out:
            subprocess.run([str(v) for v in cmd],cwd=str(repo),env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    run([sys.executable,src/'choose_unet.py','--run',a.run],'unet_optimization_selection')
    run([sys.executable,src/'evaluate.py','--run',a.run,'--mode','select'],'validation_selection')
    processes=[];handles=[]
    for name in ['unet','segformer']:
        cmd=[sys.executable,str(src/'train_torch.py'),'--run',str(a.run),'--model',name,'--initial',str(a.run/name/'best.pt'),'--threads','1']
        for mode in ['test','train']:
            # Independent model scoring jobs; four one-thread processes share the4-core quota.
            log=(a.run/(name+'_'+mode+'_evaluation.log')).open('w');handles.append(log)
            processes.append(subprocess.Popen(cmd+['--evaluate',mode],cwd=str(repo),env=env,stdout=log,stderr=subprocess.STDOUT))
    for process in processes:assert process.wait()==0,'Read evaluation log for failure'
    for handle in handles:handle.close()
    docker_env=env.copy()
    for key in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:docker_env.pop(key,None)
    def container_path(path):return str(Path('/work')/Path(path).resolve().relative_to(repo))
    prefix=['docker','--host=unix:///var/run/docker.sock','run','--rm','--user','1000:1000','--cpus','4',
        '--mount','type=bind,src='+str(repo)+',dst=/work','-w','/work','-e','OMP_NUM_THREADS=4','-e','KMP_BLOCKTIME=0','b3-tf112-mkl:local',
        'python',container_path(src/'train_tf.py'),'--run',container_path(a.run),'--vendor',container_path(a.vendor),
        '--initial',container_path(a.run/'deeplab/best_weights_fp16.npz'),'--threads','4']
    for mode in ['test','train']:
        with (a.run/('deeplab_'+mode+'_evaluation.log')).open('w') as out:
            subprocess.run(prefix+['--evaluate',mode],env=docker_env,stdout=out,stderr=subprocess.STDOUT,check=True)
    import json
    winner=json.loads((a.run/'selection.json').read_text())['winner']
    if winner=='deeplab':
        with (a.run/'selected_upper.log').open('w') as out:subprocess.run(prefix+['--evaluate','upper'],env=docker_env,stdout=out,stderr=subprocess.STDOUT,check=True)
    else:
        run([sys.executable,src/'train_torch.py','--run',a.run,'--model',winner,'--initial',a.run/winner/'best.pt','--threads','4','--evaluate','upper'],'selected_upper')
    run([sys.executable,src/'evaluate.py','--run',a.run,'--mode','report'],'root_report')
    if not (a.run/'label_geometry_audit.json').exists():
        run([sys.executable,src/'label_audit.py','--run',a.run],'label_geometry_audit')
    run([sys.executable,src/'package.py','--run',a.run,'--output',a.output],'root_package')
    print('ROOT_STAGE_COMPLETE; temporal stage remains unstarted',flush=True)
if __name__=='__main__':main()
