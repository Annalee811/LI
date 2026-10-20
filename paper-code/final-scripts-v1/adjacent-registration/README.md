# Adjacent image registration

Python3.12. Install the pinned dependencies. The official RoMa implementation and license are included under `vendor/`. Model weights are downloaded by the upstream loader and checked against `model_provenance.json`.

```bash
python -m pip install -r requirements.txt
python run.py prepare --raw-root /path/to/input-data --work experiment
python run.py render --work experiment
python run.py confidence --work experiment
```

Input files are identified by SHA256, so their directory names do not matter. `registration_parameters.json` contains the exact coordinate transforms and observation domains; `expected_outputs.json` pins reconstructed output files. Reconstruction uses these supplied parameters. RoMa computes adjacent dense correspondences for cumulative confidence and effective overlap. Missing dates remain absent. Input data and neural weights are separate from this code archive. CPU/GPU changes may affect neural fields; supplied geometric reconstruction is checksum-verified.

The output includes adjacent correspondence fields, cumulative confidence, geometric/effective overlap masks, registered RGB and numerical area/confidence tables. Cumulative score is the product of the smaller forward/reverse score at each transported step. Effective overlap requires score >=0.5, displacement <=15 reference pixels and cycle error <=10 reference pixels at every step, within the same observation domain. Geometric coverage and these operational scores are not independently measured landmark accuracy. Retained `review_required` flags are part of the supplied parameters.
