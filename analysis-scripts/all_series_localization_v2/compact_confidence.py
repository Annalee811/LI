"""Archive exact float32 cumulative C without its auxiliary matrix duplicate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from render_results import sha

def compact(work):
    root=work/'deliverable';dest=work/'archive_numeric_confidence';manifest=work/'archive_numeric_manifest.json';records=json.loads(manifest.read_text()) if manifest.exists() else {}
    for f in sorted(root.glob('*/confidence/*_cumulative_confidence_float32.npz')):
        if not (f.parent/'confidence_summary.json').exists():continue
        rel=f.relative_to(root).as_posix();source_sha=sha(f)
        if rel in records and records[rel]['source_NPZ_SHA256']==source_sha and (dest/rel).exists():continue
        with np.load(f) as z:c=z['confidence'];assert c.dtype==np.float32
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        # Outer ZIP compresses this uncompressed NPZ once; values stay exact.
        np.savez(target,confidence=c)
        with np.load(target) as z:np.testing.assert_array_equal(z['confidence'],c)
        records[rel]=dict(source_NPZ_SHA256=source_sha,archive_NPZ_SHA256=sha(target),retained_fields=['confidence'],dtype='float32',shape=list(c.shape),C_values_SHA256=hashlib.sha256(c.tobytes()).hexdigest(),C_values_bit_exact=True,omitted_auxiliary_field='bottleneck_confidence matrix; mean retained in CSV and matrix reproducible')
    manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2));return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();r=compact(a.work);print('LOSSLESS_C_CONFIDENCE_BUNDLES',len(r),flush=True)
