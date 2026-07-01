# Mesh QC

Standalone HPC-first quality control for generated m2m head meshes.

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

- `found_meshes.csv`: discovered mesh paths and inferred mesh/subject/repeat labels.
- `qc_summary.csv`: objective mesh metrics for every mesh.
- `qc_flags.csv`: only non-OK meshes.
- `renders/meshes/*.png`: per-mesh render tiles for meshes that passed QC loading.
- `mosaics/all_mesh_wall.png`: combined wall across all loadable meshes.

ROI labels are retained as optional CSV metadata when the path contains ROI run folders, but they are not used by default for progress labels or wall generation. To also write separate ROI wall mosaics, pass:

```bash
--roi-walls
```

Use `--skip-renders` for a fast CSV-only dry run or on systems without PyVista display support.

Rendering does not require PyVista by default. `--renderer auto` tries PyVista first and falls back to a pure Pillow/NumPy software renderer. To avoid PyVista entirely, pass:

```bash
--renderer pillow
```

Progress is shown during discovery, QC, rendering, and mosaic creation. By default, `--progress auto` uses `tqdm` progress bars when `tqdm` is installed and falls back to plain text otherwise.

Force a mode with:

```bash
--progress tqdm
--progress text
--progress none
```

On very large HPC trees, discovery may be the slowest first step. In text progress mode, tune the discovery heartbeat with:

```bash
--discovery-progress-seconds 2
```

In text progress mode, tune per-mesh QC/render progress with:

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
