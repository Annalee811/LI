"""Upload immutable RGB/archive objects using standard Git LFS, then verify bytes.

Only normal GitHub LFS endpoints are used. Temporary signed transfer URLs and
their authorization headers are never printed or saved.
"""
import argparse,hashlib,json,os,subprocess,urllib.parse,urllib.request
from pathlib import Path
from render_results import sha

def object_cache(path,storage):
    oid=sha(path);dest=storage/'objects'/oid[:2]/oid[2:4]/oid;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        try:os.link(path,dest)
        except OSError:
            import shutil
            shutil.copy2(path,dest)
    assert dest.stat().st_size==path.stat().st_size and sha(dest)==oid
    return dict(oid=oid,size=path.stat().st_size,source=str(path))

def upload(objects,repo,storage):
    ids=sorted({r['oid'] for r in objects})
    subprocess.run(['git','-c','lfs.storage='+str(storage),'-c','lfs.concurrenttransfers=4','lfs','push','--object-id','origin',*ids],cwd=repo,check=True)

def verify_object(oid,size):
    body={'operation':'download','transfers':['basic'],'objects':[{'oid':oid,'size':size}]}
    request=urllib.request.Request('https://github.com/Annalee811/LI.git/info/lfs/objects/batch',data=json.dumps(body).encode(),headers={'Content-Type':'application/vnd.git-lfs+json','Accept':'application/vnd.git-lfs+json'},method='POST')
    with urllib.request.urlopen(request,timeout=45) as r:obj=json.load(r)['objects'][0]
    assert 'error' not in obj,obj.get('error');action=obj['actions']['download'];url=urllib.parse.urlparse(action['href']);assert url.scheme=='https'
    assert url.hostname in {'github-cloud.githubusercontent.com','github-cloud.s3.amazonaws.com','lfs.github.com'}
    request=urllib.request.Request(action['href'],headers=action.get('header',{}));h=hashlib.sha256();n=0
    with urllib.request.urlopen(request,timeout=60) as response:
        status=response.status
        while b:=response.read(4*1024*1024):n+=len(b);h.update(b)
    assert status==200 and n==size and h.hexdigest()==oid
    return dict(object_SHA256=oid,bytes=n,download_SHA256=h.hexdigest(),transfer_host=url.hostname,HTTP_status=status,full_download_verified=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--storage',type=Path,required=True);p.add_argument('--kind',choices=['rgb','zip'],required=True);p.add_argument('--verify',action='store_true');p.add_argument('--write-pointers',action='store_true');a=p.parse_args();a.storage.mkdir(parents=True,exist_ok=True)
    records=[]
    if a.kind=='rgb':
        paths=sorted((a.work/'deliverable').glob('*/RGB/T*_independent_stitched_RGB.png'));assert len(paths)==66
        for f in paths:
            r=object_cache(f,a.storage);r['position']=f.parent.parent.name;r['period']=f.name[:2];r['canonical_path']='roma-results/all-series-localization-v2/RGB/'+r['position']+'/'+r['period']+'_independent_RGB.png';records.append(r)
    else:
        r=object_cache(a.work/'all_series_original_and_stitched_RGB_v2.zip',a.storage);r['canonical_path']='roma-results/all-series-localization-v2/'+Path(r['source']).name;records=[r]
    upload(records,a.repo,a.storage)
    if a.verify:
        proofs=[verify_object(r['oid'],r['size']) for r in records];(a.work/(a.kind+'_LFS_download_verification.json')).write_text(json.dumps(proofs,indent=2))
    if a.write_pointers:
        for r in records:
            f=a.repo/r['canonical_path'];f.parent.mkdir(parents=True,exist_ok=True);f.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+r['oid']+'\nsize '+str(r['size'])+'\n')
    (a.work/('lfs_'+a.kind+'_objects.json')).write_text(json.dumps(records,indent=2));print('LFS_UPLOADED',a.kind,len(records),'unique',len({r['oid'] for r in records}),flush=True)
if __name__=='__main__':main()
