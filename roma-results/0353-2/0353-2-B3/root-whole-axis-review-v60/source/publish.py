"""Publish only reviewed editor deliverables, preserving every older commit."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('/workspace/LI')
OUT=ROOT/'results/B3_root_whole_trace_v60_20261026'
REPO='repos/Annalee811/LI'
BRANCH='annotation-restore-v50-delivery'
PARENT='a674cd20babb6e1f5c30a370a0d19cdad70a0a8a'
PREFIX='roma-results/0353-2/0353-2-B3/root-whole-axis-review-v60'
EDITOR='roma-results/0353-2/0353-2-B3/original-annotation-restore-v50/annotate_restore_fixed_lite.html'


def api(path,obj=None,method=None):
    command=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:command+=['--input','-']
    if method:command+=['--method',method]
    result=subprocess.run(command,cwd=ROOT,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr[:2000])
    return json.loads(result.stdout)


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    browser=json.loads((OUT/'browser_verification.json').read_text())
    assert browser['passed'] and browser['tested_editor_SHA256']==hashlib.sha256((OUT/'annotate.html').read_bytes()).hexdigest()
    assert json.loads((OUT/'review_data_verification.json').read_text())['passed']
    assert not api('')['private'], 'Do not mislabel a private repository link as public'
    manifest=[]
    for name in ['annotate.html','review_project.json','review_data_verification.json','browser_verification.json','README.md']:
        manifest.append((OUT/name,PREFIX+'/'+name))
    for p in sorted((OUT/'source').glob('*')):
        if p.name=='atlas.py':continue
        if p.is_file():manifest.append((p,PREFIX+'/source/'+p.name))
    manifest.append((OUT/'annotate.html',EDITOR))
    statepath=OUT/'published_blob_state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    entries=[];records=[]
    for local,dest in manifest:
        data=local.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if state.get(blob)!=len(data):
            response=api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))
            assert response['sha']==blob
            # Verify each blob before advancing the public editor branch.
            back=api('git/blobs/'+blob);assert base64.b64decode(back['content'])==data
            state[blob]=len(data);write(statepath,state)
        entries.append(dict(path=dest,mode='100644',type='blob',sha=blob))
        records.append(dict(path=dest,bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),blob=blob))
        print('VERIFIED_BLOB',dest,len(data),flush=True)
    remote=api('git/ref/heads/'+BRANCH)['object']['sha']
    pubpath=OUT/'publication_state_final.json'
    publication=json.loads(pubpath.read_text()) if pubpath.exists() else {}
    manifest_hash=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    if publication.get('commit') and remote==publication['commit'] and publication.get('manifest_SHA256')==manifest_hash:
        commit=remote
    else:
        assert remote==PARENT, 'Branch changed: preserve concurrent edits; do not force-push'
        tree=api('git/trees',dict(base_tree=api('git/commits/'+PARENT)['tree']['sha'],tree=entries))['sha']
        commit=api('git/commits',dict(message='Only reviewed visible-axis numbers; selectable unassigned whole-root paths and user-defined four-period identities',tree=tree,parents=[PARENT]))['sha']
        write(pubpath,dict(parent=PARENT,branch=BRANCH,commit=commit,manifest_SHA256=manifest_hash))
        api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    # Confirm the committed paths resolve to exactly the already verified blobs.
    tree=api('git/trees/'+commit+'?recursive=1')
    assert not tree.get('truncated'), 'Do not silently trust an incomplete path listing'
    paths={p['path']:p['sha'] for p in tree['tree'] if p['type']=='blob'}
    assert all(paths[r['path']]==r['blob'] for r in records)
    links=dict(commit=commit,branch=BRANCH,verification='Public GitHub API blob read-back and committed-path verification',
        editor_url='https://raw.githubusercontent.com/Annalee811/LI/'+BRANCH+'/'+EDITOR,
        versioned_editor_url='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/annotate.html',
        json_url='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/review_project.json',
        files=records)
    write(OUT/'download_links.json',links)
    print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':main()
