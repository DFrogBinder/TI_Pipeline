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
  - Historical utility for copying the selected `m2m_*` directories into a
    fixed-mesh cache layout.
  - Retained for traceability and exploratory geometry-control work.
  - Do not use it to create a final quantitative paper-analysis root unless the
    resulting outputs are explicitly labelled as freshly simulated
    direct-corrected data and are not mixed with scaled-rescale outputs.

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

## Retired Fixed-Median Dataset Workflow

The following root was audited and removed from the final workflow:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset`

Audit classification: `direct_corrected`.

The root was invalid for final figures because it combined two different
numerical pathways:

- `remesh`: symlinks into the scaled current-corrected root.
- `fixed_mesh`: fresh direct corrected-current simulations at pair 1 =
  `0.002 A`, pair 2 = `0.001588656 A`.

That mixed-method comparison made fixed_mesh values fall outside the remesh
distribution for some subjects. It should remain a forensic/audit result, not a
paper-analysis result.

## Interpretation Boundary

This package is a figure-selection record. It should be described as
geometry-preserving, not as ground truth. The reported field and ROI values for
the final Repeatability paper should come from one corrected root consistently.
Under the current decision, that root is the scaled current-corrected root.
