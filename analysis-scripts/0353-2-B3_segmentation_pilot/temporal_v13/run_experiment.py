"""Run the complete revised-label experiment from the supplied native data."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--publish',type=Path);a=p.parse_args();src=Path(__file__).resolve().parent
    os.environ.setdefault('MPLCONFIGDIR','/tmp/B3_v13_mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/B3_v13_cache')
    def call(name,*args):subprocess.run([sys.executable,str(src/name),*[str(v) for v in args]],check=True)
    call('baseline_references.py','--run',a.run,'--previous',a.previous)
    call('run_all.py','--data',a.run/'data','--roots',a.roots,'--previous',a.previous,'--output',a.run,'--epochs',24,'--samples',48)
    call('select_chains.py','--run',a.run,'--roots',a.roots,'--previous',a.previous)
    call('deliver.py','--run',a.run,'--roots',a.roots,'--products')
    call('measure_final_fit.py','--run',a.run,'--previous',a.previous)
    call('evaluate_reference_versions.py','--run',a.run,'--previous',a.previous)
    for script in ['compare.py','audit_chains.py']:call(script,'--run',a.run,'--previous',a.previous)
    call('verify.py','--run',a.run,'--roots',a.roots);call('export_focus.py','--run',a.run)
    call('figures.py','--run',a.run)
    for script in ['figures_compare.py','figures_chains.py']:call(script,'--run',a.run,'--previous',a.previous)
    call('figures_revision.py','--run',a.run,'--original-data',a.run/'data/original_reference')
    if a.publish:call('package.py','--run',a.run,'--roots',a.roots,'--previous',a.previous,'--publish',a.publish)
if __name__=='__main__':main()
