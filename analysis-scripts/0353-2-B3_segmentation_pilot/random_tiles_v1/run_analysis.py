"""One-command evaluation, both RGB/root-only comparisons and publication figures."""
import argparse,os,subprocess,sys,tempfile,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();source=Path(__file__).resolve().parent
    cache=Path(tempfile.mkdtemp(prefix='B3_random_publication_cache_'));env=os.environ.copy()
    (cache/'matplotlib').mkdir();(cache/'xdg').mkdir()
    env['MPLCONFIGDIR']=str(cache/'matplotlib');env['XDG_CACHE_HOME']=str(cache/'xdg')
    try:
        for script in ['analyze.py','comparison.py','large_local_comparison.py','figures.py','report.py','verify_outputs.py']:
            subprocess.run([sys.executable,str(source/script),'--run',str(a.run)],check=True,env=env)
    finally:shutil.rmtree(cache)

if __name__=='__main__':main()
