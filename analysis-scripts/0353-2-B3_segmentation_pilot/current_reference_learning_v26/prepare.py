"""Prepare a frozen, container-readable current-reference view without editing inputs."""
import argparse,os,shutil
from pathlib import Path
import common as c


def verify(folder,rows):
    for row in rows:
        for key,hash_key in [('RGB_path','RGB_sha256'),('mask_path','mask_sha256')]:
            p=folder/Path(row[key]).relative_to('data')
            assert p.is_file() and c.sha(p)==row[hash_key],str(p)


def main():
    parser=argparse.ArgumentParser()
    for key in ['run','source','trajectory']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();reference=args.source/'revised_report/data'
    rows=c.csvread(reference/'manifest.csv');original={r['image_id']:r for r in c.csvread(args.source/'data/manifest.csv')}
    for row in rows:
        previous=original[row['image_id']]
        assert row['split']==previous['split'] and row['RGB_sha256']==previous['RGB_sha256']
        if row['split']=='train':assert row['mask_sha256']==previous['mask_sha256']
    verify(reference,rows)
    assert {s:sum(r['split']==s for r in rows) for s in ['train','val','test']}==dict(train=200,val=20,test=24)
    allocation={}
    for row in rows:allocation.setdefault(row['tile_id'],set()).add(row['split'])
    assert all(len(value)==1 for value in allocation.values())
    args.run.mkdir(parents=True,exist_ok=True)
    if (args.run/'data').exists():
        assert c.sha(args.run/'data/manifest.csv')==c.sha(reference/'manifest.csv');verify(args.run/'data',rows)
    else:
        destination=args.run/'current_data_files'
        if not destination.exists():
            temporary=args.run/'current_data_files.preparing'
            assert not temporary.exists(),'Inspect incomplete data preparation before retrying'
            shutil.copytree(reference,temporary,symlinks=False);verify(temporary,rows);os.replace(temporary,destination)
        verify(destination,rows)
        assert not any(p.is_symlink() for p in destination.rglob('*'))
        temporary=args.run/'data.next';temporary.symlink_to(destination.name,target_is_directory=True);os.replace(temporary,args.run/'data')
    for model in c.MODELS:
        (args.run/model).mkdir(exist_ok=True)
        history=c.csvread(args.trajectory/model/'history.csv')
        assert len(history)==30 and [int(r['epoch']) for r in history]==list(range(1,31))
    prep=dict(reference='User-confirmed current annotation reference',reference_manifest_SHA256=c.sha(reference/'manifest.csv'),training_targets_unchanged_from_published_final_run=True,validation_targets_current_reference=True,train_views=200,validation_views=20,test_views=24,source_original_RGB_images=4,all_dates_at_one_coordinate_in_same_split=True,source_run=str(args.source),trajectory_run=str(args.trajectory),replay_epochs=30,phase_boundary_epoch=21,phase1='Replay20epoch current-target training',phase2='Restore reported final current-target checkpoint and replay its10epoch continuation',learning_rates='Frozen recorded per-epoch rates; monitoring does not select new reported weights',historical_validation_measurements_recovered=False,record_type='Actual newly measured training-trajectory replay',save_model_state_after_every_epoch=True,old_reference_curves_in_figures=False,external_independent_test_images=0,reported_models_unchanged=True,current_data_view='Byte-identical materialized files without host-absolute symlinks',provenance_archive=str(args.source/'revision/provenance.json'))
    path=args.run/'preparation.json'
    if path.exists():assert c.read(path)['reference_manifest_SHA256']==prep['reference_manifest_SHA256']
    else:c.js(path,prep)
    print('CURRENT_REFERENCE_INPUTS_VERIFIED_AND_READY',flush=True)


if __name__=='__main__':main()
