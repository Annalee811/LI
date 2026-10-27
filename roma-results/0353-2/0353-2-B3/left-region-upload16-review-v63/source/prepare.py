"""Visually reviewed, explicit per-segment proposals; never propagate old IDs blindly."""
import copy,csv,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;O=R/'results/B3_left_region_review_v63_20261027'
# Each selection was checked against the four native RGB crops. These remain
# assistant proposals, not human-confirmed biological identities.
PLANS=[
 dict(label='0004-1',mother='0004',segments=[[65],[104,127],[129],[117]],evidence='T1基部距0004现有轨迹约2.8px；四期同一向下转右的细根轴；未与邻近平行根85/139混并。'),
 dict(label='0007-7',mother='0007',segments=[[124],[222],[225],[214]],evidence='T1基部距0007现有轨迹约5px；沿四期同一向右下弯曲侧根保留编号；上部遮挡不补画。'),
 dict(label='0031',segments=[[139,167,168],[219,230],[213],[204]],evidence='中央向右下的细根轴，四期可见片段同向对应；交叉处未吸收横向/向左下的157/165等根段。'),
 dict(label='0032',segments=[[164],[227],[226],[216]],evidence='左中部向右下弯曲的独立可辨根轴；四期对应，不与旁边竖直166系列混号；母根归属尚未确定。'),
 dict(label='0033',segments=[[166],[233],[235],[]],evidence='左中部竖直细根前三期对应；T4仅有零星像素，不能可靠判断同一根，T4对应尚未确定。'),
 dict(label='0034',segments=[[170],[237,673],[238,643],[220,679]],evidence='从左边界进入、向右弯曲的基期根轴，后期沿同一曲线延续；不按长度增加另编新根。'),
 dict(label='0035',segments=[[224,239,243],[352,368,372,387,393],[348],[335,361]],evidence='下方从左边界进入、缓向右下的同一根轴；气泡/土壤造成中间缺段，仅串联已存在的对应片段；分支230/231等仍未定。'),
 dict(label='0036',segments=[[268,289],[408,453],[416],[414]],evidence='左下偏右向右下的斜根轴四期对应；不合并旁边向左下的315/455系列；母根尚未确定。'),
 dict(label='0037',segments=[[275],[502],[678],[511]],evidence='最下部从左下边界向右下穿过视野的根轴；各期边界附近可辨部分沿用编号，遮挡不补画。'),
 dict(label='0038',segments=[[282],[463,527,468],[455,679],[471]],evidence='底部较竖直、略向左弯的根轴，四期对应；未把附近斜向283/733等根段混入。'),
 dict(label='0039',segments=[[315],[455],[478],[472]],evidence='底部向右上延伸的弯曲根轴，各期沿用基期身份；未吸收邻近垂直471或另一条平行334根。'),
 dict(label='0040',segments=[[334],[492],[496],[486]],evidence='底部另一条平行斜根，与0039保持独立身份；跨期均在右侧对应位置，不按距离近直接合并。'),
]
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 raw=(O/'source_progress_16.json').read_bytes();original=json.loads(raw);d=copy.deepcopy(original);st=d['root_structure_annotation'];hs={h['root_id']:h for h in st['root_track_hierarchy']};used={h['label'] for h in hs.values() if h.get('annotation_status_v60')!='unassigned'}
 next_id=max(int(s['root_id'][1:]) for f in st['periods'] for s in f['segments'])+1;changed=[];proposals=[]
 original_formal={k:copy.deepcopy(h) for k,h in hs.items() if h.get('annotation_status_v60')!='unassigned'}
 for p in PLANS:
  assert p['label'] not in used;used.add(p['label']);rid='R'+str(next_id);next_id+=1;parent=next((h for h in hs.values() if h['label']==p.get('mother')),None);family=parent['family_root_id'] if parent else rid
  selected={};observed=[]
  for f,indices in zip(st['periods'],p['segments']):
   ss=[next(v for v in f['segments'] if v['index']==i) for i in indices];selected[f['period']]=[v['id'] for v in ss]
   if ss:observed.append(f['period'])
   for v in ss:
    old=hs[v['root_id']];assert old.get('annotation_status_v60')=='unassigned',(p['label'],v['id'],old['label'])
    assert all(0<=q[0]<=500 and 180<=q[1]<1108 for q in v['points']),(p['label'],v['id'],'outside screenshot subregion')
    changed.append(dict(period=f['period'],segment_id=v['id'],old_root_id=v['root_id'],new_root_id=rid,proposed_label=p['label']))
    v.update(root_id=rid,display_label=p['label'],parent_root_id=parent['root_id'] if parent else None,family_root_id=family,branch_kind='existing' if parent else 'independent',review_status='pending')
    v['assistant_identity_proposal_v63']=dict(source='visual_four_period_review',human_confirmed=False)
  h=dict(root_id=rid,label=p['label'],parent_root_id=parent['root_id'] if parent else None,family_root_id=family,branch_kind='existing' if parent else 'independent',review_status='pending',parentage_confirmed=False,annotation_status_v60='assistant_proposed_v63',hidden_technical_label=False,candidate_path_only=False,first_seen_period='T1',observed_periods=observed,birth_role_v58='existing_lateral_proposed' if parent else 'baseline_axis_parentage_unresolved',independent_new_confirmed_v58=False,assistant_proposal_v63=dict(evidence=p['evidence'],parentage='proposed_from_visible_T1_attachment' if parent else 'unresolved',human_confirmed=False,selected_segments=selected,not_a_claim_of_independent_new_root=True))
  st['root_track_hierarchy'].append(h);hs[rid]=h
  st['root_tracks'].append(dict(root_id=rid,first_seen_period='T1',observed_periods=observed,identity_status='assistant_proposed_T1_axis',review_status='pending',biological_identity_confirmed=False,display_label=p['label'],parent_root_id=h['parent_root_id'],family_root_id=family,branch_kind=h['branch_kind']))
  st['root_lifetime_registry_v58'].append(dict(root_id=rid,label=p['label'],parent_root_id=h['parent_root_id'],birth_role=h['birth_role_v58'],identity_retained_through_T4=True,visibility_is_not_identity=True,states=[dict(period=f['period'],root_id=rid,label=p['label'],parent_root_id=h['parent_root_id'],observed=f['period'] in observed,status='observed_proposed_same_identity' if f['period'] in observed else 'identity_retained_match_not_confirmed',match_confirmed=False) for f in st['periods']]))
  proposals.append(dict(root_id=rid,label=p['label'],mother=p.get('mother'),observed_periods=observed,selected_segments=selected,evidence=p['evidence'],human_confirmed=False))
 # Retire only empty technical U handles after regrouping their native pieces.
 # The editor requires one live hierarchy entry for each actually present ID.
 active={v['root_id'] for f in st['periods'] for v in f['segments']}
 retired=[h for h in st['root_track_hierarchy'] if h['root_id'] not in active]
 assert all(h.get('annotation_status_v60')=='unassigned' for h in retired)
 st['retired_unassigned_candidates_v63']=dict(hierarchy=copy.deepcopy(retired),root_tracks=[v for v in st['root_tracks'] if v['root_id'] not in active],lifetimes=[v for v in st['root_lifetime_registry_v58'] if v['root_id'] not in active],reason='原有轨迹已全部归入本次补标组，仅退役空的技术U标识；原记录留存，不删除轨迹或人工定义')
 st['root_track_hierarchy']=[h for h in st['root_track_hierarchy'] if h['root_id'] in active]
 st['root_tracks']=[v for v in st['root_tracks'] if v['root_id'] in active]
 st['root_lifetime_registry_v58']=[v for v in st['root_lifetime_registry_v58'] if v['root_id'] in active]
 assert {h['root_id'] for h in st['root_track_hierarchy']}==active
 assert all(not h.get('parent_root_id') or h['parent_root_id'] in active for h in st['root_track_hierarchy'])
 # Existing evidence is preserved; links involving changed identities are marked
 # for review, never silently asserted as human-confirmed correspondence.
 changes={(c['period'],c['segment_id']) for c in changed}
 for link in st['cross_period_links']:
  if (link['previous_period'],link['previous_segment']) in changes or (link['current_period'],link['current_segment']) in changes:
   link.update(review_status='pending',kind='identity_link_needs_review');link['note_v63']='局部补标初稿，需要人工确认跨期对应'
 st['human_reviewed']=False;st['status']='proposal_under_review'
 unresolved=[dict(region='左下两条靠近并交叉的斜根，约x=0..260,y=958..1108',reason='穿越关系和侧根归属尚未确定；保留完整候选供用户定义，未自动编号'),dict(region='左中部0033在T4的对应',reason='只有零星短像素，不能确认，未补画'),dict(region='0001下方纵向细根与其余遮挡区域',reason='后期连续性或连接归属尚未确定，不按缺失直接判新生')]
 st['regional_review_v63']=dict(source_upload='root_annotation_progress (16).json',source_SHA256=hashlib.sha256(raw).hexdigest(),screenshot_region_bbox=[0,180,750,1108],proposal_count=len(proposals),proposals=proposals,unresolved=unresolved,original_manual_definitions_preserved=True,pixel_layers_unchanged=True,original_segment_geometry_and_owner_maps_unchanged=True,training_started=False,all_proposals_require_human_review=True)
 for key,h in original_formal.items():assert hs[key]==h
 for old,new in zip(original['images'],d['images']):assert old==new
 for old,new in zip(original['root_structure_annotation']['periods'],st['periods']):
  assert old['nodes']==new['nodes'] and old['owner_map_rle']==new['owner_map_rle']
  for a,b in zip(old['segments'],new['segments']):
   assert a['id']==b['id'] and a['points']==b['points'] and a['index']==b['index'] and a['length_px']==b['length_px']
   if (old['period'],a['id']) not in changes:assert a==b
  for k in old:
   if k!='segments':assert old[k]==new[k]
 (O/'review_project.json').write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n');(O/'proposal_plan.json').write_text(json.dumps(dict(proposals=proposals,unresolved=unresolved,changed_segments=changed),ensure_ascii=False,indent=2)+'\n')
 with (O/'proposal_summary.csv').open('w',newline='') as fd:
  w=csv.writer(fd);w.writerow(['proposed_label','mother_proposed','T1_segments','T2_segments','T3_segments','T4_segments','human_confirmed','evidence'])
  for p in proposals:w.writerow([p['label'],p['mother'] or '尚未确定',*[','.join(p['selected_segments'][period]) or '尚未确定' for period in ['T1','T2','T3','T4']],False,p['evidence']])
 report=dict(passed=True,source_SHA256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),preserved_user_definitions=sum(h.get('annotation_status_v60')=='user_defined' for h in original_formal.values()),preserved_total_formal_definitions=len(original_formal),assistant_proposals=len(proposals),retired_empty_technical_U_handles=len(retired),proposals_with_four_observed_periods=sum(len(p['observed_periods'])==4 for p in proposals),proposed_laterals=sum(bool(p['mother']) for p in proposals),changed_segment_identity_assignments=len(changed),pixel_layers_byte_value_exact=True,all_segment_points_width_owner_maps_nodes_lengths_exact=True,all_original_formal_definitions_exact=True,no_original_formal_segment_changed=True,all_original_manual_cut_records_exact=True,all_changes_within_screenshot_region=True,unknown_T4_correspondence_not_invented=True,unresolved_crossings_not_renumbered=True,review_project_SHA256=hashlib.sha256((O/'review_project.json').read_bytes()).hexdigest())
 (O/'data_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
