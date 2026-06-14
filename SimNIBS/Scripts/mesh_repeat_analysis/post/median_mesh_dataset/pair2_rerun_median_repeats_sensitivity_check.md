# Median-Representative Remesh Repeat Selection

This table selects, for each subject, the remesh repeat whose requested metric is closest to the subject-specific remesh median.
If the requested metric is unavailable for a subject, the selected row records the fallback metric and selection basis.
For 40 repeats, the arithmetic median is usually between the 20th and 21st sorted values, so the lower and upper median-bracketing repeats are also reported.

## Selected Repeats

| Subject | Selected repeat | Metric | Basis | Target median | Selected value | Delta | Lower bracket | Upper bracket |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| sub-CC120120 | repeat_004 | median_roi | requested_metric | 0.208185099 | 0.208348095 | +1.630e-04 | repeat_033 | repeat_004 |
| sub-CC122620 | repeat_003 | median_roi | requested_metric | 0.243397463 | 0.243319079 | -7.838e-05 | repeat_003 | repeat_036 |
| sub-CC222496 | repeat_002 | median_roi | requested_metric | 0.176146716 | 0.176008493 | -1.382e-04 | repeat_002 | repeat_022 |
| sub-CC321506 | repeat_001 | median_roi | requested_metric | 0.245763399 | 0.245277777 | -4.856e-04 | repeat_001 | repeat_019 |
| sub-CC410182 | repeat_005 | median_roi | requested_metric | 0.240250006 | 0.240102842 | -1.472e-04 | repeat_005 | repeat_037 |
| sub-CC420075 | repeat_002 | median_roi | requested_metric | 0.195924621 | 0.195688516 | -2.361e-04 | repeat_002 | repeat_005 |
| sub-CC510534 | repeat_022 | median_roi | requested_metric | 0.161381714 | 0.161268845 | -1.129e-04 | repeat_022 | repeat_036 |
| sub-CC520209 | repeat_002 | median_roi | requested_metric | 0.208851498 | 0.208698772 | -1.527e-04 | repeat_002 | repeat_021 |
| sub-CC711128 | repeat_015 | median_roi | requested_metric | 0.171208799 | 0.171134159 | -7.464e-05 | repeat_015 | repeat_033 |
| sub-CC721418 | repeat_006 | median_roi | requested_metric | 0.182471149 | 0.182400614 | -7.053e-05 | repeat_006 | repeat_023 |

## Rerun Note

Use `selected_m2m_dir` from the CSV as the source mesh directory for a median-representative fixed-mesh rerun. The `selected_mesh_path` column records the corresponding `.msh` file inside that directory. If the selected mesh directory is not visible on the current machine, run the selection on the cluster output filesystem or use the same inferred path there.

CSV: `median_representative_remesh_repeats.csv`
