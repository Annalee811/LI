"""Verified UTF8 HTML blob delivery, avoiding unnecessary base64 expansion."""
import base64,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_editable_upload17_v69_20261028';HERE=Path(__file__).resolve().parent
REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/editable-upload17-offline-v69'
ALIAS='roma-results/0353-2/0353-2-B3/original-annotation-restore-v50/annotate_restore_fixed_lite.html'
def api(path,obj=None,method=None):
    cmd=['gh','api',REPO+('/'+path if path else '')]
    if obj is not None:cmd+=['--input','-']
    if method:cmd+=['--method',method]
    p=subprocess.run(cmd,cwd=ROOT,input=None if obj is None else json.dumps(obj,ensure_ascii=False),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:1600])
    return json.loads(p.stdout)
def raw_blob(blob):
    p=subprocess.run(['gh','api',REPO+'/git/blobs/'+blob,'-H','Accept: application/vnd.github.raw+json'],cwd=ROOT,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode()[:1600])
    return p.stdout
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    test=json.loads((OUT/'browser_verification.json').read_text());build=json.loads((OUT/'build_verification.json').read_text())
    assert test['passed'] and build['passed']
    assert test['tested_HTML_SHA256']==build['offline_html_sha256']==hashlib.sha256((OUT/'annotate.html').read_bytes()).hexdigest()
    assert hashlib.sha256((OUT/'root_annotation_progress_17.json').read_bytes()).hexdigest()==build['source_sha256']
    assert not api('')['private']
    for name in ['build.py','verify.mjs','publish.py']:shutil.copy2(HERE/name,OUT/'source'/name)
    names=['annotate.html','annotate_online.html','root_annotation_progress_17.json','build_verification.json','browser_verification.json','README.md','source/build.py','source/verify.mjs','source/publish.py']
    mapping=[(name,PREFIX+'/'+name) for name in names]+[('annotate.html',ALIAS)]
    cachepath=OUT/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];files=[]
    for name,dest in mapping:
        data=(OUT/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if cache.get(blob)!=len(data):
            # HTML/JSON/scripts are UTF8 text. Sending text keeps the offline HTML
            # request below28MiB instead of expanding the whole file another33%.
            result=api('git/blobs',dict(encoding='utf-8',content=data.decode('utf-8')))
            assert result['sha']==blob
            back=raw_blob(blob)
            if back!=data:
                decoded=json.loads(back);assert base64.b64decode(decoded['content'])==data
            cache[blob]=len(data);js(cachepath,cache)
        entries.append(dict(path=dest,mode='100644',type='blob',sha=blob));files.append(dict(path=dest,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob=blob))
        print('VERIFIED',name,len(data),flush=True)
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];manifest=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();statepath=OUT/'publication_state.json';prev=json.loads(statepath.read_text()) if statepath.exists() else {}
    if prev.get('commit')==head and prev.get('manifest_sha256')==manifest:commit=head
    else:
        tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(tree=tree,parents=[head],message='Save exact uploaded17 as offline editable HTML with original PNGs; preserve all labels and arbitrary-cut editing'))['sha']
        js(statepath,dict(parent=head,commit=commit,manifest_sha256=manifest));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
    assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit
    remote=api('git/trees/'+commit+'?recursive=1');assert not remote.get('truncated');paths={v['path']:v['sha'] for v in remote['tree'] if v['type']=='blob'};assert all(paths[r['path']]==r['blob'] for r in files)
    base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/'
    links=dict(commit=commit,editable_html=base+'annotate.html',smaller_online_html=base+'annotate_online.html',backup_JSON=base+'root_annotation_progress_17.json',stable_editor='https://raw.githubusercontent.com/Annalee811/LI/'+BRANCH+'/'+ALIAS,verification='GitHub API raw-byte readback, SHA256, and committed path verification',files=files)
    js(OUT/'download_links.json',links);js(OUT/'completion.json',dict(state='complete',commit=commit,offline=True,browser_checks=len(test['checks']),source_sha256=build['source_sha256'],all_annotations_and_native_geometry_preserved=True))
    print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
