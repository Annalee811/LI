"""Reject reusing frozen non-Q1 geometry with changed original RGB files."""
import argparse,json
from pathlib import Path
from render_results import sha

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
registry=json.loads((a.work/'registry_non_Q1.json').read_text())
for pos,entry in registry.items():
    files={p.stem:p for p in (a.work/'inputs/raw'/pos).iterdir() if p.is_file()}
    for r in entry['regions']:
        expected=r['evidence']['input_sha256']
        assert all(sha(files[t])==h for t,h in expected.items()),(pos,'Frozen geometry source changed: recompute localization.')
print('PASS all frozen non-Q1 regions match their exact original input hashes')
