"""Publish a separate annotation revision and inference review; preserve historical paths."""
import base64
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/LI')
OUT = ROOT / 'results/B3_branch_continuity_review_v85_20261028'
REV = ROOT / 'results/B3_reference_local_correction_v84_20261028'
HERE = Path(__file__).resolve().parent
REPO = 'repos/Annalee811/LI'
BRANCH = 'annotation-restore-v50-delivery'
PARENT = 'roma-results/0353-2/0353-2-B3/'
PREFIX = PARENT + 'U-Net-branch-aware-continuity-review-v85'
REFPREFIX = PARENT + 'U-Net-user-confirmed-reference-correction-v84'


def api(path, obj=None, method=None):
    cmd = ['gh', 'api', REPO + ('/' + path if path else ''), '-H', 'Cache-Control: no-cache']
    if obj is not None:
        cmd += ['--input', '-']
    if method:
        cmd += ['--method', method]
    p = subprocess.run(cmd, cwd=ROOT, input=None if obj is None else json.dumps(obj), capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[:1200])
    return json.loads(p.stdout)


def js(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    assert json.loads((OUT / 'verification.json').read_text())['passed']
    assert json.loads((OUT / 'viewer_browser_verification.json').read_text())['passed']
    assert json.loads((REV / 'verification.json').read_text())['passed']
    assert not api('')['private']
    for p in HERE.glob('*.py'):
        shutil.copy2(p, OUT / 'source' / p.name)
    (REV / 'source').mkdir(exist_ok=True)
    shutil.copy2(HERE.parent / 'reference_local_correction_v84/correct_and_evaluate.py', REV / 'source/correct_and_evaluate.py')
    names = [p.name for p in OUT.glob('*.csv')] + [p.name for p in OUT.glob('*.json') if not p.name.startswith(('published', 'publication', 'download'))]
    names += ['RESULTS.md', 'Root_by_root_review_offline.html']
    for dirname, pattern in [('full', '*.png'), ('changes', '*.png'), ('figures', '*'), ('source', '*.py')]:
        names += [dirname + '/' + p.name for p in sorted((OUT / dirname).glob(pattern)) if p.is_file()]
    refnames = ['root_annotation_progress_24_user_corrected_T2_new_root.json', 'reference_correction_audit.json', 'verification.json', 'evaluation_results.json', 'RESULTS.md', 'U_Net_Dice_before_after_reference_correction.csv', 'T2_reference_changed_pixels.png', 'T2_User_corrected_reference_and_unchanged_UNet.tif', 'T2_User_corrected_reference_and_unchanged_UNet_preview.png', 'source/correct_and_evaluate.py']
    refnames += ['corrected_reference/' + p.name for p in sorted((REV / 'corrected_reference').glob('*.png'))]
    cachepath = OUT / 'published_blob_cache.json'
    cache = json.loads(cachepath.read_text()) if cachepath.exists() else {}
    entries, files = [], []
    for folder, prefix, selected in [(OUT, PREFIX, sorted(set(names))), (REV, REFPREFIX, refnames)]:
        for name in selected:
            data = (folder / name).read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if cache.get(blob) != len(data):
                if name.endswith(('.html', '.json')):
                    payload = dict(encoding='utf-8', content=data.decode('utf-8'))
                else:
                    payload = dict(encoding='base64', content=base64.b64encode(data).decode())
                assert api('git/blobs', payload)['sha'] == blob
                assert base64.b64decode(api('git/blobs/' + blob)['content']) == data
                cache[blob] = len(data); js(cachepath, cache)
            entries.append(dict(path=prefix + '/' + name, mode='100644', type='blob', sha=blob))
            files.append(dict(path=prefix + '/' + name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), blob=blob))
            print('VERIFIED', prefix.split('/')[-1], name, len(data), flush=True)
    head = api('git/ref/heads/' + BRANCH + '?nonce=' + str(time.time_ns()))['object']['sha']
    tree = api('git/trees', dict(base_tree=api('git/commits/' + head)['tree']['sha'], tree=entries))['sha']
    commit = api('git/commits', dict(tree=tree, parents=[head], message='Review all four U-Net fields with separate mother-extension and new-lateral identities; preserve model weights, annotate one explicitly confirmed reference error, and report conditional Dice and unresolved cases'))['sha']
    js(OUT / 'publication_state.json', dict(parent=head, commit=commit))
    response = api('git/refs/heads/' + BRANCH, dict(sha=commit, force=False), method='PATCH')
    assert response['object']['sha'] == commit
    committed = api('git/trees/' + commit + '?recursive=1')
    assert not committed.get('truncated')
    paths = {v['path']: v['sha'] for v in committed['tree'] if v['type'] == 'blob'}
    assert all(paths[v['path']] == v['blob'] for v in files)
    base = 'https://raw.githubusercontent.com/Annalee811/LI/' + commit + '/'
    js(OUT / 'download_links.json', dict(commit=commit, base=base + PREFIX + '/', reference_base=base + REFPREFIX + '/', files=files, verification='GitHub API exact-byte readback and immutable committed-tree verification; raw host is not reachable inside this runtime.'))
    js(OUT / 'publication_complete.json', dict(state='complete', commit=commit, files=len(files)))
    print('PUBLISHED', commit, flush=True)


if __name__ == '__main__':
    main()
