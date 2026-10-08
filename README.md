# Same-FOV temporal registration results

These archives use T1 as the fixed 2550 x 2273 reference frame.

- `*_same_fov_core.zip` contains the complete registered image, transform,
  raw RoMa warp, support mask, and metrics.
- `*_same_fov_visuals.zip` contains alpha/difference overlays, checkerboards,
  change maps, and match visualizations.

T2 is supported by a broad RoMa solution (`verified_roma`). T3 is exported
under the user-provided same-location/full-frame acquisition constraint and is
marked `prior_only_unverified`: RoMa's visually supported candidate covered
only about 42.6% of the frame. This distinction is preserved in
`metrics.json`, `transform.json`, and `roma_support_mask.png`.
