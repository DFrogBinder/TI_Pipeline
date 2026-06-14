# Repeatability Fixed-Median Mesh Dataset

This package defines the median-representative remesh repeats to use for the
Repeatability paper figures.

The selected repeats come from the scaled current-corrected analysis, not from
the pair2-rerun analysis. This is intentional: the scaled correction preserves
the original repeat geometry and applies only the deterministic pair-2 current
amplitude correction. Pair2-rerun is retained here as a sensitivity check,
because it regenerates the second-pair electrode-augmented simulation mesh and
therefore changes the median representative repeat for every subject.

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
  - Self-contained Stanage/HPC script for copying the selected `m2m_*`
    directories into a new fixed-mesh cache layout.

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

## HPC Dataset Creation

On Stanage, the authoritative selection CSV should also exist at:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv`

Use that path directly on the HPC filesystem. Copy or rsync
`hpc_seed_fixed_median_mesh_dataset.py` to the HPC if it is not already there.

Recommended root for the seeded dataset:

`/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset`

Dry run first:

```bash
python hpc_seed_fixed_median_mesh_dataset.py --selection-csv /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv --new-root /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset --dry-run
```

Seed the dataset:

```bash
python hpc_seed_fixed_median_mesh_dataset.py --selection-csv /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv --new-root /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset
```

If rerunning after a partial copy:

```bash
python hpc_seed_fixed_median_mesh_dataset.py --selection-csv /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv --new-root /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset --overwrite
```

Validate the seeded mesh cache count:

```bash
find /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset -path "*/fixed_mesh/mesh_cache/*/anat/.mesh_ready.json" | wc -l
```

Expected count: `10`.

Validate the mesh count:

```bash
find /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset -path "*/fixed_mesh/mesh_cache/*/anat/m2m_*/*.msh" | wc -l
```

Expected count: `10`.

## Config Creation

After seeding, create a repeatability config that points at the new root. From
the TI_Pipeline repository on Stanage:

```bash
python -c 'import json; from pathlib import Path; src=Path("SimNIBS/Scripts/mesh_repeat_analysis/paired_repeatability_experiment.json"); dst=Path("/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset_config.json"); data=json.loads(src.read_text()); data["experiment_root"]="/mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset"; dst.write_text(json.dumps(data, indent=2)+"\n"); print(dst)'
```

Validate the config/root after simulations or after any copied outputs are
present:

```bash
python SimNIBS/Scripts/mesh_repeat_analysis/validate_experiment_outputs.py --config /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset_config.json --strict --output-json /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset_validation.json --output-csv /mnt/parscratch/users/cop23bi/current-repair/repeatability_scaled_fixed_median_mesh_dataset_validation.csv
```

## Interpretation Boundary

This dataset is a figure-selection and fixed-median-cache dataset. It should be
described as geometry-preserving, not as ground truth. If a final quantitative
analysis uses pair2-rerun values, the reported field and ROI values should still
come from the pair2-rerun root; this selection only fixes which original
repeat/mesh is treated as the representative median geometry for figures.
