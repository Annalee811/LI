"""Rebuild all v9 paper tables, ten figure groups and editable report from inputs."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True);os.environ.setdefault('MPLCONFIGDIR','/tmp/B3_v9_manuscript_mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/B3_v9_cache')
 for name,args in [('analyze.py',['--inputs',a.inputs,'--output',a.output/'tables']),('figures.py',['--inputs',a.inputs,'--tables',a.output/'tables','--output',a.output/'figures']),('report.py',['--run',a.output])]:subprocess.run([sys.executable,str(here/name),*[str(v) for v in args]],check=True)
if __name__=='__main__':main()
