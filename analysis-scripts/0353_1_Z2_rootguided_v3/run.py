"""Full reproducible root-guided region recheck for exactly 0353-1-Z2."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args();source=Path(__file__).resolve().parent;support=source/'support'
 if not support.exists():support=source.parent/'all_series_localization_v2'
 os.environ.setdefault('MPLCONFIGDIR','/tmp/LI_Z2_plot_cache');os.environ['PYTHONPATH']=str(support/'vendor')+os.pathsep+str(support)+os.pathsep+os.environ.get('PYTHONPATH','')
 if Path('/workspace/.cache/torch/hub/checkpoints/roma_outdoor.pth').exists():os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
 pins=json.loads((source/'input_sha256.json').read_text())
 for period,digest in pins.items():assert hashlib.sha256(next(a.raw.glob(period+'.*')).read_bytes()).hexdigest()==digest,period
 def run(f,*args):subprocess.run([sys.executable,str(f),*map(str,args)],check=True,env=os.environ)
 run(source/'prepare.py','--work',a.work,'--raw',a.raw);run(source/'descriptor_step.py','--work',a.work)
 # These cached full-model fields are reused only if input SHA256 matches.
 run(support/'infer_roma.py','--work',a.work/'search','--positions','0353-1-Z2')
 run(support/'infer_roma.py','--work',a.work/'search','--positions','0353-1-Z2','--extra-pair','T1_to_T3')
 run(source/'prepare_alt.py','--work',a.work);run(support/'infer_roma.py','--work',a.work/'tight_gray','--positions','0353-1-Z2','--extra-pair','T2_to_T3')
 run(source/'coarse_bridge.py','--work',a.work);run(source/'geometry.py','--work',a.work)
 run(support/'render_results.py','--work',a.work,'--registry',a.work/'registry.json','--positions','0353-1-Z2')
 run(support/'infer_roma.py','--work',a.work,'--positions','0353-1-Z2','--view','posed')
 run(support/'render_results.py','--work',a.work,'--registry',a.work/'registry.json','--positions','0353-1-Z2','--confidence')
 run(source/'report.py','--work',a.work);run(source/'verify.py','--work',a.work);run(source/'package.py','--work',a.work)
if __name__=='__main__':main()
