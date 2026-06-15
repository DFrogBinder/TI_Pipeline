# Repeatability Median-Representative Mesh Selection

This package defines the median-representative remesh repeats to use for the
Repeatability paper figures.

The selected repeats come from the scaled current-corrected analysis, not from
the pair2-rerun analysis. This is intentional: the scaled correction preserves
the original repeat geometry and applies only the deterministic pair-2 current
amplitude correction. Pair2-rerun is retained here as a sensitivity check,
because it regenerates the second-pair electrode-augmented simulation mesh and
therefore changes the median representative repeat for every subject.

Important current decision:

- The final Repeatability quantitative figures should use the scaled
  current-corrected root consistently.
- Do not mix scaled-rescale remesh outputs with newly simulated direct-corrected
  fixed-mesh outputs.
- The previous `repeatability_scaled_fixed_median_mesh_dataset` workflow is
  retired for final paper figures. It produced a mixed-method analysis root:
  remesh values came from the scaled root, while fixed_mesh values came from
  fresh direct corrected-current simulations.
- The selected-repeat CSV remains valid as a representative-geometry selection
  record; it should not be treated as an instruction to regenerate a new final
  quantitative fixed-median dataset.

## Files

- `scaled_geometry_preserving_median_repeats.csv`
  - Authoritative selection for paper figures.
  - Source: `repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv`.
  - Metric: `median_roi`.
  - Condition: `remesh`.
- `scaled_geometry_preserving_median_repeats.md`
  - Human-readable version of the authoritative selection.
- `pair2_rerun_median_repeats_sensitivity_check.csv`
  - Recomputed median representative repeats from the pair2-rerun analysis.
  - This is not the paper-figure selection source.
- `scaled_vs_pair2_rerun_median_repeat_comparison.csv`
  - Subject-level comparison showing that the selected repeat differs for all
    10 subjects between scaled and pair2-rerun.
- `dataset_metadata.json`
  - Machine-readable decision record.
- `hpc_seed_fixed_median_mesh_dataset.py`
  - Utility for copying the selected `m2m_*` directories into a fixed-mesh
    cache layout.
  - For final fixed-median presentation work, run it with
    `--seed-repeat-workspaces` so each fixed repeat workspace is pre-populated
    with links to the selected scaled remesh repeat's anatomy/reference inputs.
  - Do not use `m2m_*`-only seeding by itself for the final fixed-median
    paper-analysis root.

## Authoritative Selected Repeats

| Subject | Selected repeat |
| --- | --- |
| `sub-CC120120` | `repeat_021` |
| `sub-CC122620` | `repeat_016` |
| `sub-CC222496` | `repeat_013` |
| `sub-CC321506` | `repeat_009` |
| `sub-CC410182` | `repeat_004` |
| `sub-CC420075` | `repeat_003` |
| `sub-CC510534` | `repeat_009` |
| `sub-CC520209` | `repeat_011` |
| `sub-CC711128` | `repeat_022` |
| `sub-CC721418` | `repeat_005` |

## HPC Usage For Final Figures

On Stanage, the authoritative selection CSV should also exist at:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv`

Use that path directly on the HPC filesystem. The final quantitative analysis
should read values from:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled`

Recommended setup:

```bash
cd /users/cop23bi/Repos/TI_Pipeline
export BASE=/mnt/parscratch/users/cop23bi/current-repair
export SCALED_ROOT=$BASE/repeatability_scaled
export SELECTION_CSV=$SCALED_ROOT/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv
```

Sanity-check the selection file:

```bash
python -c 'import csv, os; rows=list(csv.DictReader(open(os.environ["SELECTION_CSV"]))); print("rows", len(rows)); [print(r["subject"], r["selected_repeat_tag"]) for r in rows]'
```

Expected selected repeats:

```text
sub-CC120120 repeat_021
sub-CC122620 repeat_016
sub-CC222496 repeat_013
sub-CC321506 repeat_009
sub-CC410182 repeat_004
sub-CC420075 repeat_003
sub-CC510534 repeat_009
sub-CC520209 repeat_011
sub-CC711128 repeat_022
sub-CC721418 repeat_005
```

Check that each selected remesh output exists in the scaled root:

```bash
python -c 'import csv, os; root=os.environ["SCALED_ROOT"]; rows=list(csv.DictReader(open(os.environ["SELECTION_CSV"]))); missing=[]; [missing.append(f"{sub} {rep}") for r in rows for sub,rep in [(r["subject"], r["selected_repeat_tag"])] if not os.path.exists(os.path.join(root, f"{sub}_repeatability", "remesh", "repeats", rep, sub, "anat", "SimNIBS", "Output", sub, "TI.msh"))]; print("missing", len(missing)); [print(x) for x in missing]'
```

Expected: `missing 0`.

## Fixed-Median Presentation Dataset

The standard `repeatability_scaled` root is internally consistent, but its
`fixed_mesh` condition uses the original fixed mesh, not the selected median
remesh geometry. Therefore, the fixed-mesh points are expected to lie within the
remesh distribution, but they do not have to be centered on the selected remesh
median.

The remaining presentation-only data generation step is therefore not another
current correction. The pair-2 field has already been corrected in
`repeatability_scaled`. The missing control is geometry/reference selection:
the fixed-mesh condition needs to use each subject's selected median remesh
repeat as its fixed geometry/reference.

The required workflow is:

1. Read the selected median remesh repeats from
   `scaled_geometry_preserving_median_repeats.csv`.
2. Seed the fixed-mesh dataset from the selected repeat's scaled-root geometry
   and anatomy/reference inputs, not from `m2m_*` alone:

   ```bash
   python SimNIBS/Scripts/mesh_repeat_analysis/post/median_mesh_dataset/hpc_seed_fixed_median_mesh_dataset.py \
     --selection-csv "$SELECTION_CSV" \
     --new-root "$MEDIAN_FIXED_ROOT" \
     --seed-repeat-workspaces
   ```

3. Rerun only the fixed_mesh condition with the intended corrected currents.
4. Keep the remesh side from the existing `repeatability_scaled` root.
5. Re-run only the downstream analysis/figure generation needed for the
   presentation figure.

This produces a median-fixed presentation dataset while staying inside the
scaled corrected-current methodology. It should not introduce legacy-current
simulation or mix scaled remesh outputs with a different correction method.

## Retired Fixed-Median Dataset Workflow

The following root was audited and removed from the final workflow:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset`

Audit classification: `m2m_only_seeded_fixed_median`.

The root was invalid for final figures because it combined two different
workflow assumptions:

- `remesh`: symlinks into the scaled current-corrected root.
- `fixed_mesh`: newly generated simulations from a cache seeded only with the
  selected `m2m_*` directory while common anatomical inputs were still linked
  from the base source root.

That m2m-only seeding was not a clean reconstruction of the selected median
repeat as the fixed reference. It should remain a forensic/audit result, not a
paper-analysis result.

## Interpretation Boundary

This package is a figure-selection record. It should be described as
geometry-preserving, not as ground truth. The reported field and ROI values for
the final Repeatability paper should come from one corrected root consistently.
Under the current decision, that root is the scaled current-corrected root.
