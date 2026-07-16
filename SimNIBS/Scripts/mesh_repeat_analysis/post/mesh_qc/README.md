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

For long runs, use the submission helper:

```bash
bash mesh_repeat_analysis/hpc_scripts/submit_mesh_qc.sh
```

That helper explicitly exports safe log paths into the batch job so cluster-level `LOG_DIR` environment variables cannot break startup. It writes Slurm stdout/stderr as `slurm-<jobid>.out` and `slurm-<jobid>.err` in the mesh QC log directory. It also defaults the HPC batch path to `RENDERER=gmsh`, so a cluster run now fails loudly if Gmsh rendering is unavailable instead of silently producing PyVista/Pillow-style fallback images.

The raw Slurm wrapper is still available at:

```bash
sbatch mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm
```

## Full Left Hippocampus Tissue-Wall Array

The production launcher processes all ten Left Hippocampus repeats. It is a
ten-task Slurm array, with one independent task and output directory per
repeat. From the HPC repository root, submit the complete scope with one
command:

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts
sbatch mesh_repeat_analysis/hpc_scripts/run_left_hippocampus_tissue_walls_array.slurm
```

No shell exports are required. The production scope is:

- dataset: all `Left_Hippocampus_Data_01` through `Left_Hippocampus_Data_10`,
- subjects: 175 unique subjects,
- tasks: 10 repeats with 175 meshes per task, 1,750 meshes total,
- array: `1-10%10`, so all repeats may run concurrently,
- expected CHARM output when all nine tags are present: 31,500 tissue tiles and 180 walls,
- output: `/mnt/parscratch/users/cop23bi/mesh-wall/Left_Hippocampus_tissue_front_back_all_repeats/Left_Hippocampus_Data_XX`,
- persistent logs: `/mnt/parscratch/users/cop23bi/mesh-wall/logs/Left_Hippocampus_tissue_front_back_all_repeats/Left_Hippocampus_Data_XX`,
- tissue-only mode: geometry QC and ordinary whole-mesh rendering are skipped,
- front and back views at `1200 x 1200`,
- 16 allocated CPUs and at most 16 subject workers per repeat, 64 GB RAM per task, and an 8-hour limit,
- site `gmsh/4.11.1-foss-2022b` and `Xvfb/21.1.6-GCCcore-12.2.0`, with the conflicting SimNIBS module disabled,
- Python: `$HOME/.conda/envs/ti-post/bin/python`.

The 16-worker value is an upper bound. The first subject is rendered in an
isolated process to measure peak memory, and the existing memory guard lowers
the active worker count if 16 real meshes would not fit safely in 64 GB.
Parallelism is across subjects: each worker loads one large mesh once, extracts
all tissues, and generates both views. Nested tissue pools are intentionally
avoided because they would duplicate mesh loading or transfer large surface
arrays between processes without increasing the allocated CPU budget.

Tissue walls add one dependency that whole-mesh Gmsh walls do not need: the
tetrahedral tissue labels must be parsed before each tissue surface can be
rendered. Install the locally tested parser once in `ti-post` from the login
node:

```bash
$HOME/.conda/envs/ti-post/bin/python -m pip install --requirement \
  $HOME/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis/post/mesh_qc/requirements-tissue-walls.txt
```

Each array task imports this dependency and verifies exactly 175 discoverable
meshes before rendering its repeat.

The terminal connection can be closed after `sbatch` prints the job ID. Monitor
the scheduler and job output with:

```bash
squeue -j <array-job-id>
tail -f mesh_lh_tissues_<array-job-id>_1.out
tail -f /mnt/parscratch/users/cop23bi/mesh-wall/logs/Left_Hippocampus_tissue_front_back_all_repeats/Left_Hippocampus_Data_01/mesh_qc_<array-job-id>_1.log
```

The final tissue walls are written under `mosaics/tissues/` in each repeat's output root.
Each tissue has `<tissue-slug>_wall.png` for the front and
`<tissue-slug>_back_wall.png` for the back.

If Slurm stops one or more tasks, resubmit the same full array. Tissue-only mode
reuses every existing non-empty tile, so completed work is not rendered again.

```bash
sbatch mesh_repeat_analysis/hpc_scripts/run_left_hippocampus_tissue_walls_array.slurm
```

The old `run_left_hippocampus_tissue_walls_pilot.slurm` launcher remains in the
repository for compatibility and provenance. It covers only Data_01 with one
worker and is not the production launcher.

### Retired Direct-Visibility Experiment

The one-mesh direct Gmsh visibility experiment is retained only as failure
provenance and must not be submitted. It loaded `sub-CC110056`, showed physical
volume tag `5` (scalp), and attempted to write front and back PNGs from one
Gmsh process under:

```text
/mnt/parscratch/users/cop23bi/mesh-wall/gmsh_tissue_visibility_smoke/<job-id>/
```

Stanage job `10904815` produced no non-empty image after almost three hours.
Direct physical-volume rendering is therefore retired for cohort walls. The
campaign policy requires full requested scope by default; reduced smoke or
pilot runs require explicit user authorization.

Default command-line execution is tuned for large interactive HPC sessions:

- direct Python rendering defaults to `--renderer auto`, which prefers Gmsh and then falls back,
- both Slurm launchers default to `RENDERER=gmsh`, which fails loudly if native Gmsh rendering is unavailable,
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
- `qc_exception_details.csv`: per-mesh QC/read exceptions with traceback text.
- `render_exception_details.csv`: per-mesh render exceptions with traceback text.
- `render_manifest.csv`: per-render audit trail with requested renderer, actual renderer, output path, and whether the PNG was reused from a previous run.
- `renders/meshes/*.png`: per-mesh render tiles for meshes that passed QC loading.
- `mosaics/all_mesh_wall.png`: combined wall across all loadable meshes.
- `renders/tissues/<tissue-slug>/*.png`: optional front-view tissue render tiles.
- `renders/tissues_back/<tissue-slug>/*.png`: optional back-view tissue render tiles.
- `mosaics/tissues/<tissue-slug>_wall.png`: front-view cohort wall for each detected tissue tag.
- `mosaics/tissues/<tissue-slug>_back_wall.png`: back-view cohort wall for each detected tissue tag.
- `tissue_presence.csv`: tissue tags present in each mesh, independent of render success.
- `tissue_render_completeness.csv`: missing-tissue and missing-tile counts per tissue and view.
- `tissue_render_manifest.csv`: renderer and resume audit trail for tissue tiles.
- `tissue_render_exception_details.csv`: tissue extraction and rendering failures.
- `logs/mesh_qc.log`: stage-level run log.
- `logs/run_context.json`: resolved paths, arguments, Slurm environment, and CPU context for the run.
- `logs/fatal_error.txt`: written only if the pipeline aborts with an unhandled exception.

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
- refreshes path-derived subject/repeat/ROI labels from stored mesh paths when labels are unknown or a regex override is supplied,
- still skips meshes whose QC flags start with `READ_FAIL`,
- reuses existing non-empty files in `renders/meshes/`,
- parallelizes per-mesh PNG rendering with `--workers`.

This means a Gmsh render job that hits a Slurm time limit can be submitted again with the same `--out` directory. Already completed per-mesh PNGs are kept, only missing renders are generated, and mosaics are rebuilt from the combined set.

To discover meshes and generate only tissue tiles and walls, without geometry
QC or ordinary whole-mesh renders, use:

```bash
python mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py \
  --root /path/to/mesh_repeat \
  --out /path/to/tissue_wall_outputs \
  --renderer gmsh \
  --workers 0 \
  --tissue-only
```

`--tissue-only` writes `found_meshes.csv`, renders front/back tissue tiles,
builds tissue walls, and writes the tissue presence, completeness, manifest,
and exception CSVs. It reuses existing non-empty tissue tiles on resubmission.
It cannot be combined with `--render-only`, `--qc-only`, or `--skip-renders`.

Mosaic assembly is separate from per-mesh rendering. Individual renders may be complete even if `mosaics/all_mesh_wall.png` is missing. The mosaic builder uses Pillow first and falls back to ImageMagick `magick montage` or `montage` if Pillow is unavailable. Large ImageMagick mosaics are assembled in stripes to avoid one high-memory command over thousands of PNGs. On Slurm, set `IMAGEMAGICK_MODULE=<module-name>` to load a site ImageMagick module after the wrapper's `module purge`, or set `MESH_QC_MONTAGE_BIN=/path/to/magick-or-montage`.

Resumed PNGs are recorded in `render_manifest.csv` with `actual_renderer=unknown_existing`, because the rerun can only prove that the file already existed. Newly rendered PNGs record the renderer actually used, for example `gmsh`. If a previous run used `--renderer auto` and produced images you do not trust, use a fresh output directory for the Gmsh rerun or remove only the old per-mesh PNGs before rerendering.

For volumetric SimNIBS `.msh` files with tetrahedral cells, QC is run on the exterior boundary extracted from the tetrahedra. Stored triangle elements are used only when no tetrahedra are present, because stored triangles may include internal tissue interfaces and can look non-manifold even when the volume mesh is valid.

ROI labels are retained as optional CSV metadata when the path contains ROI run folders, but they are not used by default for progress labels or wall generation. To also write separate ROI wall mosaics, pass:

```bash
--roi-walls
```

To inspect every tissue included in the tetrahedral meshes, pass:

```bash
--tissue-walls
```

This reconstructs the complete boundary of each positive tetrahedral tissue tag and writes two walls per tag: the existing front view and an additional 180-degree back view. Both views use the same renderer and framing. It does not rely only on stored SimNIBS triangle interfaces, because those interfaces do not necessarily contain every side of a tissue. Known CHARM tags are given readable names; unknown positive tags are retained as `tissue_<tag>`.

`tissue_render_completeness.csv` reports front and back independently and distinguishes two problems:

- `MISSING_TISSUE`: a tag found elsewhere in the cohort is absent from one or more otherwise loadable meshes.
- `INCOMPLETE_RENDER`: the tissue exists in a mesh, but its PNG was not produced.

On the HPC, opt in by setting this control in the launcher being used:

```bash
TISSUE_WALLS_CONFIG="1"
```

For an existing QC output directory, `--render-only --tissue-walls` reuses the existing front tiles and generates the missing back tiles without rerunning geometry QC.

If old QC outputs contain `unknown_roi`, rerun `--render-only --roi-walls` after pulling the current code. Render-only will refresh ROI labels from the stored paths and rewrite the metadata CSVs before rebuilding mosaics. For explicit 4-ROI grouping, use:

```bash
--roi-regex '(Left_Hippocampus_Runs|Left_M1_Runs|Right_DLPC_Runs|Right_Thalamus_Runs)'
```

Use `--skip-renders` for a fast CSV-only dry run.

For full HPC batches, disconnected-component analysis is disabled by default because it is much slower than the degenerate-face, boundary-edge, non-manifold-edge, and bounds checks. To enable it for a smaller diagnostic run, pass:

```bash
--check-components
```

QC, whole-mesh rendering, and tissue-wall rendering are parallelized because each mesh is handled independently. For tissue walls, one worker loads one mesh once and processes the front and back views of all its tissues serially; this avoids loading a large mesh once per tissue or view and avoids nested process pools. By default `--workers 0` exposes all visible CPUs to the scheduler and lets the code scale down automatically if memory looks tight or a worker pool proves unstable. Use a smaller explicit count only when you want to cap CPU usage:

```bash
--workers 8
```

Use all visible CPUs with:

```bash
--workers 0
```

Current resource behavior:

1. Resolve the visible CPU budget from `--workers`, Slurm CPU allocation, CPU affinity, or `os.cpu_count()`.
2. Resolve a memory budget from `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU`, or cgroup limits when available.
3. Run a short isolated warmup on the first mesh(es) to estimate per-worker peak RSS.
4. Choose a stage worker count that fits both the visible CPU budget and the estimated memory budget.
5. When the active Python build supports it, use `max_tasks_per_child=1` so long QC or render runs do not keep accumulating memory inside reused worker processes.
6. If a worker pool still crashes, automatically halve the worker count and retry only the unfinished meshes.
7. If repeated pool crashes drive the worker count down to `1`, finish the remaining meshes one at a time in isolated subprocesses so that one bad mesh does not abort the whole stage.

In practice, the recommended mode on Slurm is now simply:

```bash
--workers 0
```

That treats the detected allocation as an upper bound and avoids hand-tuning worker counts.

Mosaic assembly remains serial after the per-mesh and per-tissue renders finish.

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

When `--renderer gmsh` is forced, the render stage now runs a single-mesh Gmsh preflight before starting the worker pool. If Gmsh, Xvfb, or the display backend is broken, the stage aborts immediately with one logged error instead of marking every mesh as a render failure. Gmsh subprocesses also have a bounded timeout, configurable with:

```bash
MESH_QC_GMSH_TIMEOUT_SECONDS=120
```

On Stanage, the SimNIBS module can put a bundled `gmsh` on `PATH` that fails on older nodes with a `GLIBC_2.23 not found` error. The renderer validates `gmsh -version` and skips unusable `gmsh` binaries. For Slurm runs, prefer loading a compatible Gmsh module or setting an explicit binary:

```bash
GMSH_MODULE=gmsh/<compatible-module-name>
MESH_QC_GMSH_BIN=/path/to/compatible/gmsh
```

`MESH_QC_GMSH_BIN` takes precedence when set.

For `MESH_QC_STAGE=render` with `RENDERER=gmsh`, the Slurm wrapper still loads the SimNIBS module by default because SimNIBS may be the only provider of `gmsh` on Stanage. If you explicitly configure a separate `GMSH_MODULE` or `MESH_QC_GMSH_BIN` and need to avoid SimNIBS module-stack conflicts, opt out with:

```bash
MESH_QC_SKIP_SIMNIBS_FOR_RENDER=1
```

The final mosaic wall still needs an image-composition backend. Pillow is preferred. If Pillow is not installed in the active Python environment, the code falls back to ImageMagick if `magick` or `montage` is on `PATH`. The ImageMagick fallback limits each montage call to 512 input PNGs by default and stacks temporary stripe images afterward; override this with `MESH_QC_IMAGEMAGICK_MAX_INPUTS` only if the scheduler memory limit requires it.

The Pillow path remains available as a last-resort software fallback. For smaller meshes it rasterizes triangles directly; for dense meshes it switches to a depth-based frontal preview so the output stays surface-like instead of collapsing into a sparse triangle cloud.

Progress is shown during discovery, QC, rendering, and mosaic creation. By default, `--progress auto` uses `tqdm` progress bars when `tqdm` is installed and falls back to plain text otherwise.

For debugging failed HPC runs, the code now always writes a persistent run log and structured exception CSVs into `--out`, so you do not need to rely only on transient terminal output.

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
  --repeat-regex "Data_([0-9]+)$"
```

The default repeat inference also recognizes staged dataset folders such as `Left_Hippocampus_Data_07`, yielding repeat `07`. Slurm runs can pass the same overrides with `ROI_REGEX`, `SUBJECT_REGEX`, and `REPEAT_REGEX`.
