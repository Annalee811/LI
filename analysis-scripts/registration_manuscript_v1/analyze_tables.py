import argparse,csv,json,statistics as st
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--tables',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=args.tables
def rows(n):
 with (r/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
a=rows('common_area_ratios.csv');c=rows('cumulative_confidence.csv');cat=json.loads((Path(__file__).resolve().parent/'geometry_registry_current.json').read_text());by={p:dict(position_id=p,periods=e['periods'],review_required=any(v['review_required'] for v in e['regions'])) for p,e in cat.items()}
def stats(v):return dict(n=len(v),mean=st.mean(v),sd=st.stdev(v) if len(v)>1 else None,median=st.median(v),min=min(v),max=max(v))
posmeans={p:st.mean(float(x['common_area_percent']) for x in a if x['position_id']==p) for p in by};out=dict(position_mean_area=stats(list(posmeans.values())),series={},positions=posmeans)
for series in ['0353','A017','Q1']:
 ids=[p for p in by if p.startswith(series+'-')];out['series'][series]=dict(positions=len(ids),images=sum(len(by[p]['periods']) for p in ids),four_date_positions=sum(len(by[p]['periods'])==4 for p in ids),review_required_positions=sum(by[p]['review_required'] for p in ids),position_mean_area=stats([posmeans[p] for p in ids]))
complete=[p for p in by if len(by[p]['periods'])==4];out['matched_complete_positions_cumulative']={}
for t in ['T2','T3','T4']:
 rr=[x for x in c if x['position_id'] in complete and x['target_period']==t]
 assert len(rr)==12
 out['matched_complete_positions_cumulative'][t]={k:stats([float(x[k]) for x in rr]) for k in ['mean_cumulative_confidence_on_common','effective_overlap_percent_of_common','low_cumulative_confidence_percent_of_common']}
out['examples']={p:dict(area_position_mean=posmeans[p],per_date_area={x['period']:float(x['common_area_percent']) for x in a if x['position_id']==p},cumulative=[{k:x[k] for k in ['target_period','observed_sequence','mean_cumulative_confidence_on_common','effective_overlap_percent_of_common','low_cumulative_confidence_percent_of_common']} for x in c if x['position_id']==p]) for p in ['0353-2-B3','Q1-1-Z2']}
out['review_positions']=[p for p in by if by[p]['review_required']]
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
