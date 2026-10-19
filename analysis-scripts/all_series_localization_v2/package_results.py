"""Create one ZIP, preserving source bytes and omitting redundant band copies."""
import argparse,csv,hashlib,json,re,shutil,zipfile
from pathlib import Path
from render_results import sha
from compact_confidence import compact

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--scripts',type=Path,default=Path(__file__).parent);p.add_argument('--prefill',action='store_true');a=p.parse_args();root=a.work/'deliverable'
    building=a.work/'all_series_original_and_stitched_RGB_v2.building.zip';prefill=a.work/'archive_prefill.json'
    if a.prefill:
        assert not building.exists() and not prefill.exists()
        selected=[]
        for f in sorted(root.rglob('*')):
            if not f.is_file():continue
            if f.parent.name in ['originals','localization'] or (f.parent.name=='RGB' and 'independent_stitched_RGB' in f.name):selected.append(f)
        assert len(list(root.glob('*/geometry_metadata.json')))==19
        entries=[]
        with zipfile.ZipFile(building,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for f in selected:
                name='all_series_localization_v2/'+f.relative_to(root).as_posix();z.write(f,name);entries.append(dict(archive_name=name,source=f.relative_to(root).as_posix(),sha256=sha(f)))
        prefill.write_text(json.dumps(entries,indent=2));print('PREFILLED_IMMUTABLE_IMAGES',len(entries),building.stat().st_size,flush=True);return
    validation=json.loads((root/'verification.json').read_text());assert validation['passed'] and validation['scope']=='full' and validation['original_images']==69
    numeric=compact(a.work);assert len(numeric)==48
    shutil.copy2(a.work/'archive_numeric_manifest.json',root/'archive_numeric_manifest.json')
    (root/'archive_storage_notes.json').write_text(json.dumps(dict(C_values='All cumulative C values retained bit-for-bit as float32 NPZ; compressed by outer ZIP',bottleneck='Mean retained in CSV; auxiliary matrix reproducible with included scripts',confidence_images='Native T1 16-bit PNG, stitched color previews and colorbar PDF overview retained',duplicate_images_omitted='Joined 16-bit C PNG and per-position PNG overview duplicate retained fields and PDF',originals_and_independent_RGB='All requested images preserved without numerical or color changes'),indent=2))
    scripts=root/'scripts';shutil.copytree(a.scripts,scripts,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['registry_non_Q1.json','registry_all.json','registry_Q1.json']:shutil.copy2(a.work/name,scripts/name)
    for name in ['source_manifest.json','Q1_mapping.json','download_sources.json']:
        file=a.work/'inputs'/name
        if file.exists():shutil.copy2(file,root/name)
    audit=root/'localization_evidence';audit.mkdir(exist_ok=True)
    for folder in (a.work/'q1_geometry').iterdir():
        target=audit/folder.name;target.mkdir(exist_ok=True)
        for file in folder.glob('*.json'):shutil.copy2(file,target/file.name)
    # Native original data and independent stitched RGB are present once each.
    # Regional strip duplicates remain reproducible from the included geometry.
    files=[]
    for f in sorted(root.rglob('*')):
        if not f.is_file() or f.name in ['file_manifest.json','archive_manifest.json']:continue
        if f.parent.name=='RGB' and re.fullmatch(r'T[1-4]_region_[A-Z]_RGB\.png',f.name):continue
        if f.parent.name=='confidence' and (f.name.endswith('_cumulative_confidence_16bit.png') or f.name.endswith('_cumulative_confidence_overview.png')):continue
        if '__pycache__' in f.parts or f.suffix=='.pyc':continue
        files.append(f)
    actual={f.relative_to(root).as_posix():((a.work/'archive_numeric_confidence'/f.relative_to(root)) if f.relative_to(root).as_posix() in numeric else f) for f in files}
    manifest=[dict(path=rel,bytes=f.stat().st_size,sha256=sha(f)) for rel,f in actual.items()]
    (root/'file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(root/'file_manifest.json')
    out=a.work/'all_series_original_and_stitched_RGB_v2.zip'
    if building.exists():
        old=json.loads(prefill.read_text());names={r['archive_name'] for r in old};eligible={'all_series_localization_v2/'+f.relative_to(root).as_posix() for f in files}
        assert names<=eligible
        for r in old:assert sha(root/r['source'])==r['sha256'],'An immutable prefilled image changed'
        with zipfile.ZipFile(building,'a',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for f in files:
                name='all_series_localization_v2/'+f.relative_to(root).as_posix()
                if name not in names:z.write(actual.get(f.relative_to(root).as_posix(),f),name)
        building.replace(out)
    else:
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for f in files:z.write(actual.get(f.relative_to(root).as_posix(),f),'all_series_localization_v2/'+f.relative_to(root).as_posix())
    with zipfile.ZipFile(out) as z:assert z.testzip() is None;assert len(z.namelist())==len(files)
    assert out.stat().st_size<2147483648,'GitHub LFS single-file limit; do not silently truncate or split scientific data'
    result=dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(files),ZIP_CRC_verified=True,positions=19,original_period_images=69,C_float32_values_bit_exact=True)
    (a.work/'archive_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
