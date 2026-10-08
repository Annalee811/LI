# Full-resolution, shared-coordinate temporal mosaic

T1 defines the global coordinate system and original pixel scale. RoMa overlap
between periods places each complete image in that coordinate system; content
outside the overlap is retained on an expanded 2576 x 3670 canvas.

- `full_resolution_mosaic.zip` contains a strict T1-priority mosaic, a
  seam-feathered mosaic, source/provenance map, validity mask, and transforms.
- `full_resolution_warped_layers.zip` contains the complete T1, T2, and T3
  warped layers and their masks on exactly the same 2576 x 3670 canvas.

The T1 image itself is not resized. T2 and T3 are resampled only as required by
their estimated geometric transforms. In overlap T1 has priority so temporal
root changes are not averaged into ghost structures; the feathered image blends
only narrow outer seams. The source map uses red for T1, green for areas added
by T2, and blue for areas added by T3.
