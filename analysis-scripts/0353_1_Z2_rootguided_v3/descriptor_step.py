import argparse,sys,cv2
from pathlib import Path
support=Path(__file__).parent/'support'
if not support.exists():support=Path(__file__).parent.parent/'all_series_localization_v2'
sys.path.insert(0,str(support));from localize_q1 import descriptor_search
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1);cv2.setRNGSeed(20261019)
if not (a.work/'descriptor_evidence/descriptor_search.json').exists():descriptor_search(a.work/'search/inputs/raw/0353-1-Z2',a.work/'descriptor_evidence')
