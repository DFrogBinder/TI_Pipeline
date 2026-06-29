# Mesh QC

Standalone HPC-first quality control for ROI repeat meshes.

Run from the repository root on the HPC:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/four_roi_experiment \
  --out /path/to/mesh_qc_outputs
```

By default, the tool only scans `.msh` files inside `m2m*` directories. This avoids processing every simulation output mesh in each subject/repeat folder.

To intentionally scan every `.msh` under the root, pass:

```bash
--mesh-glob "*.msh"
```

Outputs:

- `found_meshes.csv`: discovered mesh paths and inferred ROI/subject/repeat labels.
- `qc_summary.csv`: objective mesh metrics for every mesh.
- `qc_flags.csv`: only non-OK meshes.
- `renders/<roi>/*.png`: per-mesh render tiles.
- `mosaics/<roi>_wall.png`: one wall per ROI.
- `mosaics/all_roi_wall.png`: combined wall across all ROIs.

Use `--skip-renders` for a fast CSV-only dry run or on systems without PyVista display support.

Progress is printed during discovery, QC, rendering, and mosaic creation. On very large HPC trees, discovery may be the slowest first step. Tune the discovery heartbeat with:

```bash
--discovery-progress-seconds 2
```

Tune per-mesh QC/render progress with:

```bash
--progress-every 1
```

If label inference is wrong for the HPC directory layout, pass regex overrides:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/four_roi_experiment \
  --out /path/to/mesh_qc_outputs \
  --roi-regex "(Left_Hippocampus|Right_Hippocampus|M1|Pallidum)" \
  --subject-regex "(sub-CC[0-9]+)" \
  --repeat-regex "(repeat_[0-9]+)"
```
