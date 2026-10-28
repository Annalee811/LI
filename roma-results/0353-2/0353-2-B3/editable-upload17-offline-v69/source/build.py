"""Exact upload17 to editable HTML. Original PNG bytes embedded; no relabeling."""
import base64,gzip,hashlib,json,re,shutil
from pathlib import Path
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_editable_upload17_v69_20261028'
UPLOAD=Path('/workspace/attachments/f962a5bf-fbdf-4e91-a578-48e21e4cdd90/root_annotation_progress (17).json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def replace_tag(html,id,value):
    data=json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html,n=re.subn(r'(<script id="'+id+r'" type="application/json">).*?(</script>)',lambda m:m[1]+data+m[2],html,flags=re.S)
    assert n==1;return html
def main():
    OUT.mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
    raw=UPLOAD.read_bytes();doc=json.loads(raw);assert hashlib.sha256(raw).hexdigest()=='c5f82d8920e4faed263c206b4d85261dcdea99f102c4f75caf45fa6f57c4a270'
    assert len(doc['images'])==4
    shutil.copy2(UPLOAD,OUT/'root_annotation_progress_17.json')
    source=(ROOT/'results/B3_left_region_review_v63_20261027/annotate.html').read_text()
    catalogue=json.loads((ROOT/'results/B3_annotation_restore_v50_20261025/remote_image_catalogue.json').read_text())
    lookup={x['name']:x for x in catalogue};cat=[];imagechecks=[]
    for im in doc['images']:
        e=lookup[im['name']].copy();p=Path(e.pop('local_path'));assert sha(p)==e['sha256']==im['image_sha256']
        assert (e['width'],e['height'])==(im['width'],im['height'])
        dest=OUT/'assets'/e['name']
        if not dest.exists():dest.symlink_to(p)
        cat.append(e);imagechecks.append(dict(name=e['name'],sha256=sha(p),bytes=p.stat().st_size,dimensions=[e['width'],e['height']]))
    def header(html):
        return html+'''\n<script id="upload17-ui-v69">
window.upload17Ready69=(async()=>{
 await window.regionReady63;
 document.title='0353-2-B3 · (17)版可编辑标注';
 document.querySelector('h1').textContent='0353-2-B3 · 基于 (17) 版的四期可编辑标注';
 return {images:files.length,confirmed:[...rawHierarchy54().values()].filter(h=>h.annotation_status_v60==='user_defined').length};
})();
</script>\n'''
    lite=replace_tag(source,'reference-original-versions-v50',dict(original=doc));lite=replace_tag(lite,'reference-image-catalogue-v50',cat);lite=header(lite)
    (OUT/'annotate_online.html').write_text(lite)
    offlinecat=[]
    for e in cat:
        v=e.copy();v['url']='data:image/png;base64,'+base64.b64encode((OUT/'assets'/e['name']).read_bytes()).decode();offlinecat.append(v)
    offline=replace_tag(source,'reference-image-catalogue-v50',offlinecat)
    offline=replace_tag(offline,'reference-original-versions-v50',dict(original=None))
    payload=json.dumps(dict(original=doc),ensure_ascii=False,separators=(',',':')).encode()
    packed=base64.b64encode(gzip.compress(payload,compresslevel=9,mtime=0)).decode()
    decode='''
<script id="upload17-packed-v69" type="application/json">'''+json.dumps(packed)+'''</script>
<script id="upload17-decode-v69">
window.upload17PayloadReady69=(async()=>{
 const tag=document.getElementById('reference-original-versions-v50');
 // Saved edited HTML already contains the current expanded project: never overwrite it.
 if(JSON.parse(tag.textContent).original)return;
 const packed=JSON.parse(document.getElementById('upload17-packed-v69').textContent);
 const bytes=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));
 const text=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
 tag.textContent=text.replace(/<\\//g,'<\\\\/');
})();
</script>
'''
    needle='<script id="reference-original-versions-v50" type="application/json">{"original":null}</script>'
    assert offline.count(needle)==1;offline=offline.replace(needle,needle+decode)
    needle='  await window.zoom60Ready;';assert offline.count(needle)==1
    offline=offline.replace(needle,needle+'\n  await window.upload17PayloadReady69;')
    offline=header(offline);(OUT/'annotate.html').write_text(offline)
    test=lite
    for e in cat:test=test.replace(e['url'],'http://127.0.0.1:8869/assets/'+e['name'])
    (OUT/'annotate_online_test.html').write_text(test)
    # Verify both the decoded JSON and embedded original image bytes, without executing annotation logic.
    decoded=json.loads(gzip.decompress(base64.b64decode(packed)))['original'];assert decoded==doc
    for e in offlinecat:assert hashlib.sha256(base64.b64decode(e['url'].split(',',1)[1])).hexdigest()==e['sha256']
    report=dict(passed=True,source_sha256=sha(UPLOAD),backup_sha256=sha(OUT/'root_annotation_progress_17.json'),embedded_JSON_deep_equal_source=True,
        all_original_PNG_bytes_exact=True,all_four_frames_native_dimensions_unchanged=True,images=imagechecks,
        confirmed_root_IDs=sum(h.get('annotation_status_v60')=='user_defined' for h in doc['root_structure_annotation']['root_track_hierarchy']),
        new_root_labels_or_proposals_added=False,pixel_layers_geometry_widths_lengths_numbers_connections_unchanged=True,
        offline_html_bytes=(OUT/'annotate.html').stat().st_size,offline_html_sha256=sha(OUT/'annotate.html'),online_html_sha256=sha(OUT/'annotate_online.html'))
    js(OUT/'build_verification.json',report)
    (OUT/'README.md').write_text('''# Uploaded17 editable HTML (V69)

annotate.html is the recommended offline single file: exact original PNGs for all
four upper periods are embedded, together with the unchanged uploaded17 project.
Open with a recent desktop Chrome or Edge. It requires no network or image upload.
The initial JSON is gzip packed using standard browser DecompressionStream;
the saved edited HTML uses its current expanded project and never resets to17.

annotate_online.html is a smaller alternative, requiring network access to the
fixed original RGB URLs or matching local PNGs. Both contain the same annotations.
root_annotation_progress_17.json is an exact byte-for-byte original backup.

All existing human IDs, pending proposals, pixels, native paths, widths/lengths,
cross-period identities, single-owner maps, cut records and connection decisions
are retained. No GAT prediction, new proposal or automatic renumbering is added.
Existing arbitrary cutting, piece detachment, root-number editing, undo, JSON
restore and saved editable HTML controls are retained from the tested editor.

Continue editing and click 保存进度JSON and 保存可编辑HTML. Closing a browser tab
without saving does not automatically update a previously downloaded HTML file.
''')
    shutil.copy2(Path(__file__),OUT/'source/build.py')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
