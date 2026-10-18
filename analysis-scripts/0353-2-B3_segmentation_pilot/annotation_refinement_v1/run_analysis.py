"""Recompute all primary/secondary tables, native comparisons and paper figures."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    source=Path(__file__).parent
    subprocess.run([sys.executable,str(source/'make_revised_report.py'),'--run',str(a.run)],check=True)
    for name in ['analyze.py','comparison.py','figures.py','report.py']:
        subprocess.run([sys.executable,str(source/name),'--run',str(a.run/'revised_report')],check=True)
    for name in ['verify_torch.py','verify.py']:
        subprocess.run([sys.executable,str(source/name),'--run',str(a.run)],check=True)
    print('ALL_REFINEMENT_ANALYSIS_COMPLETED',flush=True)

if __name__=='__main__':main()
