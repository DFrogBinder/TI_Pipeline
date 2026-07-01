# Mesh QC

Standalone HPC-first quality control for generated m2m head meshes.

Method details are documented in [ANALYSIS.md](./ANALYSIS.md).

Run from the repository root on the HPC:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/four_roi_experiment \
  --out /path/to/mesh_qc_outputs
```

That default command runs the full pipeline:

1. discovery,
2. QC CSV generation,
3. mesh renders,
4. mosaic generation.

For long runs, a Slurm batch wrapper is available at:

```bash
sbatch mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm
```

Default execution is tuned for large interactive HPC sessions:

- rendering defaults to `--renderer auto`, which prefers Gmsh,
- renders default to `1200 x 1200`,
- QC and per-mesh rendering use all visible CPUs.

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

## Running Stages Separately

If you want to stop after QC and skip all rendering work, use:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/four_roi_experiment \
  --out /path/to/mesh_qc_outputs \
  --qc-only
```

`--skip-renders` still works, but `--qc-only` is the explicit stage name.

If you already have `found_meshes.csv` and `qc_summary.csv` in `--out` and only want to regenerate renders and mosaics, use:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/four_roi_experiment \
  --out /path/to/mesh_qc_outputs \
  --render-only
```

`--render-only`:

- skips discovery,
- skips geometry QC,
- reuses the previously written `found_meshes.csv` and `qc_summary.csv`,
- still skips meshes whose QC flags start with `READ_FAIL`,
- parallelizes per-mesh PNG rendering with `--workers`.

For volumetric SimNIBS `.msh` files with tetrahedral cells, QC is run on the exterior boundary extracted from the tetrahedra. Stored triangle elements are used only when no tetrahedra are present, because stored triangles may include internal tissue interfaces and can look non-manifold even when the volume mesh is valid.

ROI labels are retained as optional CSV metadata when the path contains ROI run folders, but they are not used by default for progress labels or wall generation. To also write separate ROI wall mosaics, pass:

```bash
--roi-walls
```

Use `--skip-renders` for a fast CSV-only dry run.

For full HPC batches, disconnected-component analysis is disabled by default because it is much slower than the degenerate-face, boundary-edge, non-manifold-edge, and bounds checks. To enable it for a smaller diagnostic run, pass:

```bash
--check-components
```

QC and per-mesh rendering are parallelized because each mesh is handled independently. By default `--workers 0` uses all visible CPUs. Use a smaller explicit count only when you want to cap CPU usage:

```bash
--workers 8
```

Use all visible CPUs with:

```bash
--workers 0
```

Mosaic assembly remains serial after the per-mesh renders finish.

Rendering now defaults to `--renderer auto`. The fallback order is:

1. Gmsh,
2. PyVista,
3. Pillow/NumPy.

To explicitly request a renderer, pass:

```bash
--renderer auto
--renderer gmsh
--renderer pyvista
--renderer pillow
```

The Gmsh path opens the original `.msh` directly, so the output is much closer to what you see when loading the mesh in Gmsh manually. If the node does not already have a `DISPLAY`, the render stage tries to start a shared Xvfb display once and lets all render workers inherit it.

The Pillow path remains available as a last-resort software fallback. For smaller meshes it rasterizes triangles directly; for dense meshes it switches to a depth-based frontal preview so the output stays surface-like instead of collapsing into a sparse triangle cloud.

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
