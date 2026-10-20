"""Fetch the exact initialization checkpoints and verify every SHA256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('initialization'));p.add_argument('--base-url',default='https://raw.githubusercontent.com/Annalee811/LI/265c23aa320746df32fcaef16cd17d5cfc76b35b/paper-code/final-scripts-v1/initialization');a=p.parse_args()
    for e in json.loads(Path(__file__).with_name('initialization.json').read_text()):
        target=a.output/e['file'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)==e['sha256']:continue
        temp=target.with_name(target.name+'.download')
        with urllib.request.urlopen(a.base_url.rstrip('/')+'/'+e['file'],timeout=180) as response,temp.open('wb') as out:
            while True:
                b=response.read(1024*1024)
                if not b:break
                out.write(b)
        assert temp.stat().st_size==e['bytes'] and digest(temp)==e['sha256'],e['file'];temp.replace(target)
        print('VERIFIED',e['file'],flush=True)
if __name__=='__main__':main()
