# Common-ROI temporal root tracking results

T1 is the fixed reference. T2 and T3 were transformed into T1 coordinates with
the original RoMa geometric solutions. All three periods were then cropped to
the largest rectangular region containing valid pixels in every period:

`T1 x=3:2548, y=1311:2203` (2545 x 892 pixels).

- `root_tracking_aligned_crops.zip` contains the three co-located, same-size
  images and crop/transform metadata.
- `root_tracking_visualizations.zip` contains T1-vs-T2 and T1-vs-T3 colored
  overlays, checkerboards, an RGB temporal composite, a side-by-side image, and
  an animated blink comparison.

In the RGB temporal composite, red is T1, green is T2, and blue is T3. Neutral
or white structures remain stable; colored structures differ by period.
