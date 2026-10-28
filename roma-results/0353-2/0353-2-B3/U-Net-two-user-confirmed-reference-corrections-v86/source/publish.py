"""Publish only the new user-confirmed reference revision and fixed-prediction scores."""
import base64
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/B3_user_confirmed_reference_corrections_v86_20261028'
PREFIX = 'roma-results/0353-2/0353-2-B3/U-Net-two-user-confirmed-reference-corrections-v86'
REPO = 'repos/Annalee811/LI'
BRANCH = 'annotation-restore-v50-delivery'


def api(path, data=None, method=None):
    cmd = ['gh', 'api', REPO + '/' + path, '-H', 'Cache-Control: no-cache']
    if data is not None: cmd += ['--input', '-']
    if method: cmd += ['--method', method]
    p = subprocess.run(cmd, cwd=ROOT, input=None if data is None else json.dumps(data), capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)


def js(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    verify = json.loads((OUT / 'verification.json').read_text()); assert verify['passed']
    for p, digest in verify['immutable_inputs_sha256'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    for p in HERE.glob('*.py'): shutil.copy2(p, OUT / 'source' / p.name)
    selected = ['RESULTS.md', 'root_annotation_progress_24_two_user_confirmed_new_roots.json', 'reference_correction_audit.json', 'verification.json', 'evaluation_results.json', 'figure_manifest.json', 'completion.json', 'U_Net_Dice_same_predictions_reference_revision.csv', 'T2_second_correction_pixels.png', 'T2_all_correction_pixels.png']
    for directory in ['corrected_reference', 'figures', 'source']:
        selected += [directory + '/' + p.name for p in sorted((OUT / directory).iterdir()) if p.is_file()]
    cachefile = OUT / 'published_blob_cache.json'; cache = json.loads(cachefile.read_text()) if cachefile.exists() else {}
    entries, files = [], []
    for name in selected:
        data = (OUT / name).read_bytes(); blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if cache.get(blob) != len(data):
            payload = dict(encoding='utf-8', content=data.decode('utf-8')) if name.endswith(('.json', '.md', '.py')) else dict(encoding='base64', content=base64.b64encode(data).decode())
            assert api('git/blobs', payload)['sha'] == blob
            assert base64.b64decode(api('git/blobs/' + blob)['content']) == data
            cache[blob] = len(data); js(cachefile, cache)
        path = PREFIX + '/' + name
        entries.append(dict(path=path, mode='100644', type='blob', sha=blob)); files.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), blob=blob))
        print('VERIFIED', name, len(data), flush=True)
    head = api('git/ref/heads/' + BRANCH + '?nonce=' + str(time.time_ns()))['object']['sha']
    tree = api('git/trees', dict(base_tree=api('git/commits/' + head)['tree']['sha'], tree=entries))['sha']
    commit = api('git/commits', dict(tree=tree, parents=[head], message='Apply two explicitly user-confirmed new-root reference corrections and rescore unchanged original and optimized U-Net predictions; no retraining'))['sha']
    js(OUT / 'publication_state.json', dict(parent=head, commit=commit))
    assert api('git/refs/heads/' + BRANCH, dict(sha=commit, force=False), 'PATCH')['object']['sha'] == commit
    committed = api('git/trees/' + commit + '?recursive=1'); assert not committed.get('truncated')
    paths = {p['path']: p['sha'] for p in committed['tree'] if p['type'] == 'blob'}
    assert all(paths[f['path']] == f['blob'] for f in files)
    js(OUT / 'download_links.json', dict(commit=commit, base='https://raw.githubusercontent.com/Annalee811/LI/' + commit + '/' + PREFIX + '/', files=files, verification='Exact GitHub API blob-byte readback and immutable commit-tree path verification.'))
    js(OUT / 'publication_complete.json', dict(state='complete', commit=commit, files=len(files)))
    print('PUBLISHED', commit, flush=True)


if __name__ == '__main__':
    main()
