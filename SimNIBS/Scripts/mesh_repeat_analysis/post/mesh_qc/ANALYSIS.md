# Mesh QC Analysis

This document describes exactly what the mesh QC step currently does.

It is intended to be a living method note. If the QC logic, defaults, outputs, or interpretation rules change, this file should be updated in the same change.

## Scope

The QC step is designed for generated SimNIBS `m2m` head meshes before downstream simulation analysis.

The CLI can now be run in separate stages:

- full pipeline,
- QC-only,
- render-only from previously written CSV outputs,
- tissue-only discovery and front/back tissue rendering.

It does not:

- run any electric field analysis,
- inspect ROI-specific simulation outputs by default,
- repair meshes,
- assess tissue shape automatically beyond tissue-presence and render-completeness checks,
- decide scientific exclusion thresholds automatically beyond the implemented flags.

## Input Discovery

By default, the CLI scans only `.msh` files inside directories whose name starts with `m2m`.

This is intended to target generated head meshes while avoiding unrelated simulation result meshes in the same dataset tree.

The discovery stage also infers metadata from the path:

- `mesh_id`: prefers the nearest `m2m*` directory name, otherwise the mesh filename stem,
- `subject`: first path component matching `sub-*` or `CC[0-9]+`,
- `repeat`: first path component matching `repeat*`, `rep*`, or `seed*`, or a staged dataset component such as `Left_Hippocampus_Data_07`,
- `roi`: optional path metadata only; not the primary grouping for QC.

If a path does not expose a repeat token, the record is labeled `unknown_repeat`.

Path-specific overrides can be supplied with `--roi-regex`, `--subject-regex`, and `--repeat-regex`. The Slurm wrapper exposes the same controls as `ROI_REGEX`, `SUBJECT_REGEX`, and `REPEAT_REGEX`.

## Surface Extraction

QC is performed on a triangular surface representation.

Current behavior:

1. Try PyVista reader if available.
2. Else try `meshio`.
3. Else try SimNIBS `mesh_io`.

For volumetric meshes:

- If tetrahedral cells are present, the QC surface is the exterior boundary extracted from the tetrahedra.
- Internal tetra faces are removed by keeping only triangle faces that occur exactly once across the tetrahedra.

For surface-only meshes:

- Stored triangle or quad cells are used directly.
- Quads are split into triangles.

When `--tissue-walls` is enabled, tissue surfaces are extracted separately from the geometry-QC surface:

1. Read positive physical tags on tetrahedral elements.
2. Select all tetrahedra belonging to one tag.
3. Enumerate their four faces and remove faces shared by two tetrahedra of that same tissue.
4. Map the surface from RAS into an upright front camera frame: screen axes are RAS `X-Z`, depth is RAS `+Y`, and projection is orthographic.
5. Map the same surface into the complementary back frame: screen axes are `-X-Z`, depth is RAS `-Y`, and projection is orthographic.

Stored triangle elements tagged `1000 + tissue_tag` are not used for this step. On real CHARM meshes those stored interfaces can be only a subset of a tissue boundary, particularly at interfaces with other tissues.

The tetrahedral-boundary path is important because stored triangle elements in a SimNIBS `.msh` can include internal tissue interfaces, which may appear non-manifold even when the exterior head surface is valid.

`--tissue-only` runs input discovery and this tissue extraction/rendering path
directly. It does not calculate geometry-QC metrics, load the generic exterior
surface, render the ordinary whole mesh, or build `all_mesh_wall.png`. It is an
additive mode; the default full, QC-only, and render-only paths are unchanged.

## Per-Mesh Geometry Checks

For each mesh surface, the QC code computes the following:

### 1. Empty Surface

Flag: `EMPTY_SURFACE`

Triggered when the derived surface has zero points or zero faces.

### 2. Degenerate Faces

Flag: `DEGENERATE_FACES`

For each triangular face, area is computed from the cross product of two triangle edges.

A face is marked degenerate when:

- `area <= 1e-12`

The reported count is `degenerate_faces`.

### 3. Boundary Edges

Flag: `BOUNDARY_EDGES`

All triangle edges are canonicalized as undirected edges and counted.

An edge that appears exactly once is treated as a boundary edge, indicating an open surface or hole.

The reported count is `boundary_edges`.

### 4. Non-Manifold Edges

Flag: `NONMANIFOLD_EDGES`

All triangle edges are counted as above.

An edge that appears more than twice is treated as non-manifold.

The reported count is `nonmanifold_edges`.

### 5. Disconnected Components

Flag: `DISCONNECTED_COMPONENTS`

This check is optional and disabled by default for full HPC runs because it is materially slower than the other checks.

When enabled with `--check-components`, connected components are computed on face adjacency through shared edges.

The reported count is `connected_components`.

When disabled, `connected_components` is set to `-1`.

### 6. Mesh Bounding Box

No direct flag is produced from the raw box alone.

For each mesh, the axis-aligned bounding-box size is recorded as:

- `x_size`
- `y_size`
- `z_size`

### 7. Bounds Outlier

Flag: `BOUNDS_OUTLIER`

After all per-mesh QC rows are computed, the tool compares `x_size`, `y_size`, and `z_size` across all meshes that completed loading.

For each dimension:

- compute `Q1`,
- compute `Q3`,
- compute `IQR = Q3 - Q1`,
- flag meshes outside `[Q1 - 4*IQR, Q3 + 4*IQR]`.

This is a dataset-relative size anomaly check, not a topology check.

## Status Logic

Each mesh row receives:

- `status = OK` if no flags were added,
- `status = FAIL` if any flag was added.

Special case:

- `READ_FAIL:<error>` is used when the mesh could not be loaded at all.

This is distinct from a geometry failure. A `READ_FAIL` means the QC logic did not reach topology analysis for that mesh.

## Rendering Logic

Rendering is separate from the geometry calculations.

Default behavior:

- renderer: `auto`,
- default render size: `1200 x 1200`,
- render only meshes that did not produce `READ_FAIL`,
- geometry-flagged meshes are still rendered,
- write a combined mosaic as `all_mesh_wall.png`,
- write ROI-specific walls only if `--roi-walls` is requested.
- write front and back walls per detected tetrahedral tissue only if `--tissue-walls` is requested.

This means a mesh with `NONMANIFOLD_EDGES` or `DEGENERATE_FACES` is still expected to appear in the wall for visual inspection.

The direct Python CLI keeps `auto` as its default renderer. The HPC Slurm launchers default to `gmsh` so batch visualization jobs fail clearly if native Gmsh rendering is unavailable instead of silently falling back to the software preview path.

Renderer selection order:

1. Gmsh,
2. PyVista,
3. Pillow/NumPy.

Gmsh renderer details:

- Gmsh is asked to open the original `.msh` directly and export a PNG.
- This is intended to keep the render visually close to what a manual Gmsh inspection would show.
- The render stage prefers one managed Xvfb display before launching parallel render workers, rather than trusting an inherited `DISPLAY` from the shell or scheduler.
- If `--renderer gmsh` is forced, the stage renders one mesh as a strict preflight before starting the worker pool.
- If that forced-Gmsh preflight fails, the stage aborts immediately and writes the first error to `render_exception_details.csv` and `render_failures.txt` instead of recording thousands of duplicate failures.
- Gmsh subprocesses are bounded by `MESH_QC_GMSH_TIMEOUT_SECONDS`, defaulting to 120 seconds.
- The selected Gmsh binary is validated with `gmsh -version`; unusable binaries, such as the SimNIBS-bundled Gmsh that fails with `GLIBC_2.23 not found` on some Stanage nodes, are skipped or rejected before rendering.
- Slurm runs can select a compatible Gmsh with `GMSH_MODULE` or bypass `PATH` entirely with `MESH_QC_GMSH_BIN`.
- On Stanage render-only Gmsh jobs skip loading `SimNIBS/4.0.1-foss-2023a` by default, because that module loads `GCCcore/12.3.0` while the compatible site `gmsh/4.11.1-foss-2022b` and `Xvfb/21.1.6-GCCcore-12.2.0` modules need `GCCcore/12.2.0`.
- The intended effect is a render that looks much closer to opening the mesh in Gmsh than the earlier software preview did.

Pillow fallback details:

- For smaller surfaces, triangles are rasterized directly.
- For dense surfaces, the renderer switches to a depth-based frontal preview built from the surface vertices plus sampled face centroids.
- This avoids the previous sparse-triangle artifact where a hard face cap could make the head look like a point cloud instead of a continuous surface.
- The software render is therefore a QC preview of the visible exterior surface, not a publication-grade mesh render.

## Parallel Execution

Each mesh is checked independently.

QC and per-mesh PNG rendering therefore parallelize naturally across worker processes. Tissue rendering uses the same process-level strategy across meshes. Inside a tissue worker, the mesh is loaded once and the front and back views of its tissue tags are processed sequentially, so workers do not repeatedly load the same large mesh or start nested pools.

For the full Left Hippocampus campaign, the outer Slurm array parallelizes the
ten repeat directories and each array task uses a process pool across its 175
subjects. Tissue/view work stays serial inside each subject worker. With 16
CPUs per array task this keeps up to 160 subject workers active across the ten
repeats while retaining one mesh load per active subject. The sampled-memory
guard can reduce workers independently in any repeat.

Current worker behavior:

- if `--workers > 0`, use that exact count,
- if `--workers == 0`, prefer `SLURM_CPUS_PER_TASK`,
- else prefer CPU affinity from `os.sched_getaffinity(0)`,
- else fall back to `os.cpu_count()`.

This is intended to respect the actual Slurm allocation instead of using the full physical node CPU count.

Current resource-management behavior on top of that:

- The resolved CPU count is treated as an upper bound, not a promise that all workers will actually be launched.
- The code tries to resolve a memory budget from `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU`, or the active cgroup memory limit.
- QC runs a small isolated warmup on the first meshes; rendering does the same on the first render task when parallelism is possible.
- That warmup captures peak worker RSS and uses it to estimate a safer worker count for the full stage.
- The current heuristic uses a 70% usable-memory budget and a 1.4x safety factor over the sampled peak RSS.
- When supported by the active Python build, parallel pools use `max_tasks_per_child=1` so worker memory does not accumulate across thousands of meshes.
- Submission backlog is bounded to roughly `2 * workers`, which avoids queueing the entire dataset in-flight at once.
- If a pool still fails with `BrokenProcessPool` or a similar worker-level crash, the stage automatically retries only the unfinished meshes with half as many workers.
- If retries eventually reduce the stage to one worker after a pool crash, the remaining meshes are executed one at a time in isolated subprocesses, allowing a single native-code failure to be marked and logged instead of aborting the entire run.

Mosaic assembly is still serial after per-mesh and per-tissue renders complete.

## Outputs

The main outputs are:

- `found_meshes.csv`: discovered meshes and inferred metadata,
- `qc_summary.csv`: one row per mesh with all metrics,
- `qc_flags.csv`: subset of rows where `status != OK`,
- `qc_exception_details.csv`: per-mesh QC/read exceptions with error type, message, and traceback,
- `render_exception_details.csv`: per-mesh render exceptions with error type, message, and traceback,
- `render_manifest.csv`: one row per successful render, including requested renderer, actual renderer, output PNG, and whether the image was reused from an earlier run,
- `renders/meshes/*.png`: per-mesh renders for non-`READ_FAIL` meshes,
- `mosaics/all_mesh_wall.png`: combined visual wall,
- `render_failures.txt`: meshes that passed QC loading but failed rendering.
- `tissue_presence.csv`: one row per mesh/tissue pair detected from tetrahedral tags,
- `tissue_render_manifest.csv`: one row per successful or resumed tissue render,
- `tissue_view_convention.json`: versioned anatomical camera convention required before existing tissue tiles can be resumed,
- `tissue_render_exception_details.csv`: tissue-load and tissue-render exceptions,
- `tissue_render_completeness.csv`: cohort counts and status per tissue and view,
- `renders/tissues/<tissue-slug>/*.png`: individual front-view tissue tiles,
- `renders/tissues_back/<tissue-slug>/*.png`: individual back-view tissue tiles,
- `mosaics/tissues/<tissue-slug>_wall.png`: front-view cohort wall per detected tissue,
- `mosaics/tissues/<tissue-slug>_back_wall.png`: back-view cohort wall per detected tissue,
- `logs/mesh_qc.log`: stage-level persistent log,
- `logs/run_context.json`: resolved runtime context including paths, arguments, CPU allocation, and key Slurm variables,
- `logs/fatal_error.txt`: written only when the run aborts with an unhandled exception.

## Stage Modes

### Full Pipeline

Default behavior:

- discover meshes,
- run QC,
- write CSV outputs,
- render loadable meshes,
- build mosaics.

### QC-Only

Triggered with `--qc-only` or the older `--skip-renders` flag.

Behavior:

- discover meshes,
- run QC,
- write `found_meshes.csv`, `qc_summary.csv`, and `qc_flags.csv`,
- stop before any PNG rendering.

### Render-Only

Triggered with `--render-only`.

Behavior:

- skip discovery,
- skip geometry QC,
- load `found_meshes.csv` and `qc_summary.csv` from `--out`,
- refresh unknown or regex-overridden path-derived subject, repeat, and ROI labels without recomputing QC,
- render only meshes whose `flags` field does not start with `READ_FAIL`,
- treat existing non-empty files in `renders/meshes/` as completed renders,
- parallelize those per-mesh renders according to `--workers`,
- write `render_manifest.csv` so the run can be audited for actual renderer use and resumed files,
- rebuild mosaics from those rendered images.
- when `--tissue-walls` is also supplied, add or resume tissue renders and rebuild tissue walls.

This mode is intended for rerendering after QC has already completed, for example when changing the renderer, image size, or wall layout.

Because Gmsh rendering is slower than the software fallbacks, this mode is also the recovery path for Slurm wall-clock limits. Re-running the same render-only command continues from the missing PNGs instead of overwriting completed ones.

Resumed files are marked with `actual_renderer=unknown_existing`, because an existing PNG does not prove how it was originally produced. Newly generated files record the renderer returned by the render backend, such as `gmsh`, `pyvista`, or `pillow`.

The render loop and mosaic assembly are separate. A Slurm log can therefore show `RENDER Complete` for all individual PNGs and still be cancelled afterwards if the job hits its wall-clock limit while building `mosaics/all_mesh_wall.png`.

Mosaic assembly is dependency-separated from rendering:

- per-mesh Gmsh renders can complete without Pillow,
- `mosaics/all_mesh_wall.png` is assembled with Pillow when available,
- if Pillow is unavailable, assembly falls back to ImageMagick `magick montage` or `montage`,
- large ImageMagick walls are assembled in stripes instead of one all-input command to reduce memory spikes,
- Slurm runs can load a site ImageMagick module with `IMAGEMAGICK_MODULE` or use a specific executable with `MESH_QC_MONTAGE_BIN`.
- if mosaic assembly fails, `render_manifest.csv` is still written and the mosaic-specific traceback is written to `mosaic_exception_details.csv`.

## Logging And Crash Diagnostics

The pipeline now writes persistent diagnostics into `--out` for every run.

Current behavior:

- Stage starts and completions are written to `logs/mesh_qc.log`.
- The resolved root/output paths, selected stage, renderer, worker request, visible CPU context, detected memory budget, and key Slurm variables are written to `logs/run_context.json`.
- Per-mesh exceptions during QC loading or QC computation are written to `qc_exception_details.csv`.
- Per-mesh exceptions during rendering are written to `render_exception_details.csv`.
- Successful render provenance is written to `render_manifest.csv`.
- If the pipeline aborts outside the per-mesh exception paths, a full traceback is written to `logs/fatal_error.txt`.
- A successful later rerun does not delete an older `logs/fatal_error.txt`; compare the timestamp and current Slurm job ID before treating it as the latest failure.

This logging is specifically intended to preserve evidence for pathing mistakes, dependency issues, virtual-display failures, and worker-pool crashes that would otherwise be visible only in transient stdout/stderr.

## Interpretation Notes

- `NONMANIFOLD_EDGES` means the exterior boundary surface contains at least one edge shared by more than two triangles.
- `BOUNDARY_EDGES` means the surface is open somewhere.
- `DEGENERATE_FACES` means at least one zero-area or near-zero-area triangle exists.
- `READ_FAIL` means the file could not be loaded or converted into a valid QC surface.

The presence of a flag does not by itself define the scientific exclusion rule. It marks meshes for review and downstream decision-making.

## Known Limitations

- Repeat labels depend on path naming conventions and may fall back to `unknown_repeat`.
- Gmsh rendering depends on a working `gmsh` executable and either a real `DISPLAY` or a working Xvfb setup.
- Dense-surface Pillow renders are still frontal depth previews rather than exact full-triangle reproductions, and are mainly a fallback path.
- `--render-only` depends on prior `found_meshes.csv` and `qc_summary.csv` outputs being present and consistent.
- Tissue extraction requires tagged tetrahedral volume elements and a readable SimNIBS or meshio backend.
- A tissue absent from every mesh cannot be inferred as expected; missing-tissue status is relative to tags present elsewhere in the loaded cohort.
- The default QC does not attempt mesh repair.
- The bounds-outlier rule is empirical and dataset-relative.
- Disconnected-component analysis is disabled by default to keep full-batch runs practical.

## Maintenance Rule

Whenever any of the following changes, this document should be updated in the same code change:

- the surface extraction logic,
- any flag definition,
- any threshold,
- the default renderer,
- worker defaults,
- rendering inclusion/exclusion rules,
- output file names or interpretation guidance.
