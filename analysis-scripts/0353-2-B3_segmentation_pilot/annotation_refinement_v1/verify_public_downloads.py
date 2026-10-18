"""Stream each public immutable ZIP and verify full byte count and SHA256."""
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
from urllib.parse import quote
import requests
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--publish',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--repository',default='Annalee811/LI');a=p.parse_args()
    prefix='roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1'
    assert len(a.commit)==40 and all(x in '0123456789abcdef' for x in a.commit)
    items=c.read(a.publish/'SHA256.json')
    def check(item):
        url='https://raw.githubusercontent.com/'+a.repository+'/'+a.commit+'/'+quote(prefix+'/'+item['file'])
        for attempt in range(3):
            with requests.get(url,stream=True,timeout=(15,60)) as response:
                if response.status_code in [429,502,503,504] and attempt<2:time.sleep(attempt+1);continue
                response.raise_for_status();h=hashlib.sha256();n=0
                for chunk in response.iter_content(1048576):h.update(chunk);n+=len(chunk)
            assert n==item['bytes'] and h.hexdigest()==item['sha256'],(item['file'],n,h.hexdigest())
            return dict(file=item['file'],url=url,bytes=n,sha256=h.hexdigest(),passed=True)
        raise RuntimeError('Download did not complete')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(check,items))
    c.js(a.publish/'public_download_verification.json',dict(data_commit=a.commit,repository=a.repository,all_passed=True,
        full_ZIP_streams_checked=len(verified),files=verified,TLS_verification_enabled=True))
    print('PUBLIC_ZIP_DOWNLOADS_VERIFIED',len(verified),sum(r['bytes'] for r in verified),flush=True)

if __name__=='__main__':main()
