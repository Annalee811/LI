from pathlib import Path
import sys,json,numpy as np
import h5py
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--position',default='0353-2-B3')
args=parser.parse_args()
p=Path.cwd()/'results/microtube_registration'/f'{args.position}_final_continuous_evaluation'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
v.configure_position(args.position)
s=json.loads((p/'continuous_evaluation.json').read_text());h,w=s['reference_size_wh'][1],s['reference_size_wh'][0];base=v.grid(h,w)
period_ids=s.get('period_ids',[1,2,3,4]);common_prefix=s.get('common_prefix','four_period')
adj=p.parent/f'{args.position}_registration_evaluation'/'evaluation_summary.json'
if adj.exists():
 a=json.loads(adj.read_text())
 s['adjacent_qc_passed_pairs']=a['automatic_qc_passed_pairs'];s['adjacent_pairs']=a['evaluated_pairs']
 s['adjacent_qc_definition']=a['automatic_qc_rules'];s['adjacent_automatic_evaluations']=a['pairs']
 s['adjacent_validation_limitations']=a['limitations']
x0,y0,x1,y1=s['largest_common_geometric_horizontal_rectangle_xyxy'];sl=np.s_[y0:y1,x0:x1]
flows=[]
for t in period_ids[1:]:
 with np.load(p/'dense_fields'/f'T1_to_T{t}_cumulative_on_T1.npz') as z:
  flows.append(v.flow_preview(z['T1_grid_to_period_xy_px'],tuple(z['period_size_wh']),base,z['geometric_mask_T1'],s['flow_display_max_T1_normalized_px'])[sl])
v.montage(flows,[d['summary']['path'] for d in s['cumulative_evaluations']],'02_cumulative_transform_composite.png',f"方向HSV；亮度log1p；幅度上限{s['flow_display_max_T1_normalized_px']:.0f} T1像素；箭头仅方向")
s['flow_display_brightness']='log1p magnitude / log1p shared p99 max; arrows normalized to fixed length indicate direction only'
(p/'continuous_evaluation.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
v.write_report(s,[d['summary'] for d in s['cumulative_evaluations']])
out=p/s.get('dense_field_filename','T1-T4_continuous_dense_fields.h5')
with h5py.File(out,'w') as hf:
 hf.attrs['coordinate_system']='T1 original pixels; zero-based pixel centres; vectors ordered (x,y)'
 hf.attrs['RGB_sampling']='registered_period_on_T1[y,x] = original_period[ T1_grid_to_period_xy_px[y,x] ]; bilinear interpolation; geometric_mask_T1 defines valid domain'
 hf.attrs['field_units']='native period pixels; to visualize displacement normalize by period/T1 sizes with half-pixel convention'
 hf.attrs['confidence_definition']=s['cumulative_confidence'];hf.attrs['effective_definition']=s['effective_definition']
 hf.attrs['reference_size_wh']=s['reference_size_wh'];hf.attrs['crop_xyxy_T1']=[x0,y0,x1,y1]
 hf.attrs['chain']=s['chain'];hf.attrs['period_ids']=period_ids;hf.attrs['missing_period_ids']=s.get('missing_period_ids',[])
 hf.attrs['cycle_export']='Cycle statistics are included in evaluation CSV/JSON; this combined field file exports the requested coordinate, confidence and mask datasets.'
 for t in period_ids[1:]:
  group=hf.create_group(f'T{t}')
  with np.load(p/'dense_fields'/f'T1_to_T{t}_cumulative_on_T1.npz') as z:
   for key in z.files:
    if key=='cycle_error_T1_px':continue
    a=z[key]
    if a.ndim>=2:group.create_dataset(key,data=a,compression='gzip',compression_opts=6,shuffle=True,fletcher32=True,chunks=True)
    else:group.create_dataset(key,data=a)
   group.attrs['path']=' -> '.join(f'T{i}' for i in period_ids[:period_ids.index(t)+1])
 with np.load(p/'dense_fields'/f'{common_prefix}_common_masks_T1.npz') as z:
  group=hf.create_group(f'{common_prefix}_common')
  for key in z.files:
   a=z[key]
   if a.ndim>=2:group.create_dataset(key,data=a,compression='gzip',compression_opts=6,shuffle=True,fletcher32=True,chunks=True)
   else:group.create_dataset(key,data=a)
with h5py.File(out,'r') as hf:
 for t in period_ids[1:]:
  with np.load(p/'dense_fields'/f'T1_to_T{t}_cumulative_on_T1.npz') as z:
   for key in z.files:
    if key=='cycle_error_T1_px':continue
    np.testing.assert_array_equal(hf[f'T{t}/{key}'][...],z[key])
 print('PASS: all combined dense fields lossless and equal to NPZ sources',flush=True)
print(out,out.stat().st_size,'bytes',flush=True)
