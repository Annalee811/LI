"""Verified immutable original checkpoints; optional reference-regeneration assets."""
import argparse,hashlib,json,tempfile,urllib.request,zipfile,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--with-original-data',action='store_true');ap.add_argument('--check-local',action='store_true');a=ap.parse_args()
    if a.check_local:
        files={'unet/best.pt':'92ddbd3a440417dc47ff21045ba23318a81176fd61c10478c0da8713ee3e928b'}
        for name,sha in files.items():
            f=a.root/'results/B3_random_tiles_v1'/name
            assert hashlib.sha256(f.read_bytes()).hexdigest()==sha
        print('LOCAL_ORIGINAL_TEACHER_SHA256_PASSED');return
    assets=json.loads(Path(__file__).with_name('starting_assets.json').read_text())
    for r in assets:
        ismodel=r['file'].endswith(('models_torch_scripts.zip','model_deeplab_scripts.zip'))
        if not ismodel and not a.with_original_data:continue
        with tempfile.TemporaryDirectory(prefix='b3-starting-') as tmp:
            path=Path(tmp)/r['file'];h=hashlib.sha256();count=0
            with urllib.request.urlopen(r['url'],timeout=120) as response,path.open('wb') as out:
                while True:
                    b=response.read(1048576)
                    if not b:break
                    out.write(b);h.update(b);count+=len(b)
            assert count==r['bytes'] and h.hexdigest()==r['sha256']
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
                for item in z.infolist():
                    pp=Path(item.filename).parts[1:]
                    if not pp or item.is_dir():continue
                    relative=Path(*pp);assert '..' not in relative.parts and not relative.is_absolute()
                    dest=a.root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as inp:
                        if dest.exists():
                            expected=hashlib.sha256(inp.read()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected,'Refuse overwrite of different existing file'
                        else:
                            with dest.open('wb') as out:shutil.copyfileobj(inp,out)
            print('VERIFIED_AND_EXTRACTED',r['file'],flush=True)

if __name__=='__main__':main()
