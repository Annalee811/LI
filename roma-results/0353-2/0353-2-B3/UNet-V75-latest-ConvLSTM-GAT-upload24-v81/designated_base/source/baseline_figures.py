"""Preserve actual historical figures and losses, no retrospective fabrication."""
import json,shutil
from prepare import OUT,BASE,MODELS,js,sha
def main():
    target=OUT/'baseline_original/figures';target.mkdir(exist_ok=True)
    records=[]
    for p in (BASE/'figures').glob('*.tif'):
        if p.name.startswith('Preview_'):continue
        dest=target/p.name
        if not dest.exists():shutil.copy2(p,dest)
        assert sha(dest)==sha(p);records.append(dict(file=str(dest.relative_to(OUT)),sha256=sha(dest),training_reference='Historical18, not relabeled as20'))
    for m in MODELS:
        import csv
        with (OUT/'baseline_original'/m/'history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
        assert [int(v['epoch']) for v in hist]==list(range(101))
    js(OUT/'baseline_preserved_figures_complete.json',dict(passed=True,original_figures=records,actual_history_epochs_each=list(range(101)),baseline_source_version='V66, originally trained on18; new20 comparison metrics computed separately'))
if __name__=='__main__':main()
