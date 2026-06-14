# Median-Representative Remesh Repeat Selection

This table selects, for each subject, the remesh repeat whose requested metric is closest to the subject-specific remesh median.
If the requested metric is unavailable for a subject, the selected row records the fallback metric and selection basis.
For 40 repeats, the arithmetic median is usually between the 20th and 21st sorted values, so the lower and upper median-bracketing repeats are also reported.

## Selected Repeats

| Subject | Selected repeat | Metric | Basis | Target median | Selected value | Delta | Lower bracket | Upper bracket |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| sub-CC120120 | repeat_021 | median_roi | requested_metric | 0.240753558 | 0.240185402 | -5.682e-04 | repeat_021 | repeat_030 |
| sub-CC122620 | repeat_016 | median_roi | requested_metric | 0.264322504 | 0.264252126 | -7.038e-05 | repeat_016 | repeat_029 |
| sub-CC222496 | repeat_013 | median_roi | requested_metric | 0.204100765 | 0.204063311 | -3.745e-05 | repeat_013 | repeat_027 |
| sub-CC321506 | repeat_009 | median_roi | requested_metric | 0.265693776 | 0.266121998 | +4.282e-04 | repeat_038 | repeat_009 |
| sub-CC410182 | repeat_004 | median_roi | requested_metric | 0.283512987 | 0.283869594 | +3.566e-04 | repeat_024 | repeat_004 |
| sub-CC420075 | repeat_003 | median_roi | requested_metric | 0.223819684 | 0.224406362 | +5.867e-04 | repeat_033 | repeat_003 |
| sub-CC510534 | repeat_009 | median_roi | requested_metric | 0.177855968 | 0.177703604 | -1.524e-04 | repeat_009 | repeat_038 |
| sub-CC520209 | repeat_011 | median_roi | requested_metric | 0.238370471 | 0.238632962 | +2.625e-04 | repeat_039 | repeat_011 |
| sub-CC711128 | repeat_022 | median_roi | requested_metric | 0.208343789 | 0.207858190 | -4.856e-04 | repeat_022 | repeat_024 |
| sub-CC721418 | repeat_005 | median_roi | requested_metric | 0.229031552 | 0.229105338 | +7.379e-05 | repeat_011 | repeat_005 |

## Rerun Note

Use `selected_m2m_dir` from the CSV as the source mesh directory for a median-representative fixed-mesh rerun. The `selected_mesh_path` column records the corresponding `.msh` file inside that directory. If the selected mesh directory is not visible on the current machine, run the selection on the cluster output filesystem or use the same inferred path there.

CSV: `median_representative_remesh_repeats.csv`
