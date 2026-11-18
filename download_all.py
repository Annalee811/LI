"""Download the published V90 artifacts with SHA256 verification; safe to restart."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,time
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('V90_downloads'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((H/'artifact_download_manifest.json').read_text(encoding='utf-8'))
for k,r in enumerate(manifest['files'],1):
 name=r['file'];assert Path(name).name==name
 dst=a.output/name
 if dst.exists()and hashlib.sha256(dst.read_bytes()).hexdigest()==r['sha256']:
  print(f"[{k}/{len(manifest['files'])}] Verified, skipping {name}");continue
 tmp=dst.with_name(dst.name+'.part')
 for attempt in range(3):
  try:
   digest=hashlib.sha256();size=0
   with urllib.request.urlopen(r['url'],timeout=120)as response,tmp.open('wb')as f:
    while True:
     chunk=response.read(1024*1024)
     if not chunk:break
     f.write(chunk);digest.update(chunk);size+=len(chunk)
   assert size==r['bytes']and digest.hexdigest()==r['sha256'],'Download hash or size mismatch'
   tmp.replace(dst);print(f"[{k}/{len(manifest['files'])}] Downloaded and verified {name}");break
  except Exception:
   if attempt==2:raise
   time.sleep(2)
print('All files downloaded and verified.')
