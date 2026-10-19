"""Fetch checksum-pinned releases and normalize original phase filenames."""
import argparse,hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path

SOURCES=[('V-.zip','98c7aa632d160f7f89714da9f68d71dddc7f28c951102dac46d8a80951ff99d2'),('Q1.zip','de74de5504474af5ef4f6e1cc28265a87ac7039032593e465ff9128336d394e2')]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def decode(i):
    if i.flag_bits&0x800:return i.filename
    try:return i.filename.encode('cp437').decode('gbk')
    except (UnicodeEncodeError,UnicodeDecodeError):return i.filename
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.work/'inputs';root.mkdir(parents=True,exist_ok=True);records=[]
    for name,expected in SOURCES:
        f=root/name;url='https://github.com/Annalee811/LI/releases/download/v-change-data/'+name
        if not f.exists():
            with urllib.request.urlopen(url,timeout=60) as response,f.open('wb') as dest:shutil.copyfileobj(response,dest)
        assert digest(f)==expected,(name,'Source checksum differs; do not silently use a new release version.')
        with zipfile.ZipFile(f) as z:
            assert z.testzip() is None
            for i in z.infolist():
                if i.is_dir():continue
                path=Path(decode(i));parts=path.parts
                if name=='V-.zip':
                    if len(parts)<3:continue
                    pos=parts[-2]
                    if pos.startswith('Q1'):continue
                else:pos=parts[-2]
                file=parts[-1];ext=path.suffix.lower()
                if ext not in ['.jpg','.jpeg','.png','.tif','.tiff']:continue
                if file.startswith('T') and file[1:2].isdigit():date=int(file[1])
                else:
                    matches=[v for k,v in {'苗期':1,'抽穗':2,'成熟':3,'枯黄':4}.items() if k in file];assert len(matches)==1,file;date=matches[0]
                target=root/'raw'/pos/('T'+str(date)+ext);target.parent.mkdir(parents=True,exist_ok=True);data=z.read(i);h=hashlib.sha256(data).hexdigest()
                if target.exists():assert digest(target)==h
                else:target.write_bytes(data)
                records.append(dict(position=pos,period=date,source_archive=name,archive_entry=str(path),sha256=h,normalized_path=str(target.relative_to(a.work))))
    assert len(records)==69 and len({r['position'] for r in records})==19
    (root/'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('PASS checksum-pinned originals:19 positions,69 images')
if __name__=='__main__':main()
