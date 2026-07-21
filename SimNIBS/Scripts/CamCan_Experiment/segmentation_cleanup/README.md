# CHARM segmentation cleanup for second-round QC

This workflow implements the scripted corrections requested after the second
round of tissue-wall review. It operates on collected flat CHARM label maps
and never modifies them in place.

## Scientific status

The CSF and skin operations reproduce the supplied Seg3D intent:

- close the cumulative WM + GM + CSF + blood envelope by 5 voxels;
- close the cumulative whole-head tissue envelope by 10 voxels; and
- rebuild the exclusive CHARM label map with the supplied tissue hierarchy.

The supplied `connectedcomponentsizefilter` commands are incomplete as an
automated removal rule. Seg3D's Connected Component Size filter creates a
component-size data image; it requires a later threshold operation to remove
small components. No threshold was included in the supplied script. This
workflow therefore makes the policy explicit:

- default: keep the largest 26-connected WM component and the largest
  26-connected cumulative WM+GM component;
- optional: `min-size`, with explicit WM and GM voxel thresholds.

The default is suitable for generating a reversible corrected-map candidate
and new tissue walls. It must not be treated as a final accepted cohort until
the corrected walls and ambiguous anatomical overlays have been reviewed.

An opt-in second candidate can also filter the cumulative
WM + GM + CSF + blood envelope to its largest 26-connected component before
closing it. This is enabled with `--csf-component-policy largest` (or the
matching launcher environment variable). The v1 default remains `none`, so
existing v1 behavior is unchanged. Filtering the cumulative envelope removes
fully disconnected islands; it does not remove bumps that remain connected to
the main envelope. Increasing the closing radius fills larger grooves and
concavities but is not equivalent to erosion-based removal of outward bumps.

## Label reconstruction

The algorithm uses CHARM tissue IDs:

| ID | Tissue |
|---:|---|
| 0 | background |
| 1 | white matter |
| 2 | gray matter |
| 3 | CSF |
| 5 | skin/scalp |
| 6 | eyes |
| 7 | compact bone |
| 8 | spongy bone |
| 9 | blood |
| 10 | muscle |
| 11 | internal air |

Unknown extended labels are preserved. Eyes, bone, blood, muscle, and air are
not discarded when the four target tissue masks are corrected.

## Non-destructive output layout

Default HPC inputs and outputs are:

```text
source:
/mnt/parscratch/users/cop23bi/charm_segmentations/maps/

corrected candidate:
/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v1/
├── maps/                         corrected NIfTI maps
├── results/                      per-subject hashes and voxel metrics
├── campaign/                     preflight and validation reports
├── collection/                   mesh-compatible collection manifest
└── logs/                         task and Slurm logs
```

Each task verifies the source hash before and after processing, writes the
corrected NIfTI atomically, reloads it, and requires the same grid and affine.
The output metadata records label volumes, every label transition, component
counts, changed-voxel fraction, parameters, and source/output SHA-256 hashes.

## Local execution

Local processing is supported when maps are already present, but transferring
the full cohort solely for correction is unnecessary. Preflight first:

```bash
python3 CamCan_Experiment/segmentation_cleanup/workflow.py preflight \
    --maps-root /path/to/original/maps \
    --output-root /path/to/corrected_v1 \
    --manifest /path/to/corrected_v1/campaign/cleanup_manifest.tsv \
    --summary /path/to/corrected_v1/campaign/preflight.json \
    --expected-subjects EXPECTED_COUNT
```

Then process the full manifest with a memory-conscious worker count:

```bash
python3 CamCan_Experiment/segmentation_cleanup/workflow.py run-all \
    --manifest /path/to/corrected_v1/campaign/cleanup_manifest.tsv \
    --workers 4
```

Validation creates the collection manifest consumed directly by
`mesh_collected_segmentations.py`:

```bash
python3 CamCan_Experiment/segmentation_cleanup/workflow.py validate \
    --manifest /path/to/corrected_v1/campaign/cleanup_manifest.tsv \
    --validation /path/to/corrected_v1/campaign/validation.tsv \
    --summary /path/to/corrected_v1/campaign/validation.json \
    --collection-manifest /path/to/corrected_v1/collection/charm_segmentation_manifest.tsv \
    --checksums /path/to/corrected_v1/collection/sha256sums.txt
```

## HPC execution

The guarded submitter discovers the flat-map cohort and requires the current
expected count of 652, hashes every source during preflight, states the
numerical scope, and submits one `%50` array plus an `afterany` collector.
No new conda environment or package installation is required on the HPC. The
submitter, correction tasks, and collector all load the existing
`SimNIBS/4.0.1-foss-2023a` module, which provides Python, NumPy, SciPy, and
NiBabel. Run the launcher from the normal base shell; do not install these
packages into the base environment.
Set `EXPECTED_SUBJECTS` explicitly only if a later audited cohort deliberately
changes that count. Defaults are a candidate profile:

- `SimNIBS/4.0.1-foss-2023a` on `sheffield`;
- 4 CPUs, 16 GB, 8 hours per correction task;
- 50 concurrent tasks and 2 retries;
- 2 CPUs, 16 GB, 8 hours for full collection validation. The previous 8 GB
  collector reached 8,390,296 KB MaxRSS, so the additional headroom is based
  on completed-run evidence.

From the repository `Scripts` directory:

```text
Scope:
  dataset: complete collected CamCan CHARM segmentation cohort
  subjects/tasks: 652
  array: 0-651%50
  corrected maps: 652
  provenance JSON files: 652
  collection manifests: 1
  execution: full discovered cohort
```

```bash
bash CamCan_Experiment/segmentation_cleanup/submit_charm_segmentation_cleanup.sh
```

For the separate CSF-radius-7/largest-component candidate, use a fresh output
root and opt in explicitly:

```bash
OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v2_csf7_lcc \
TI_CHARM_CLEANUP_CSF_RADIUS=7 \
TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY=largest \
TIME_LIMIT=08:00:00 \
COLLECTOR_MEMORY=16G \
COLLECTOR_TIME=08:00:00 \
JOB_NAME=charm_seg_cleanup_v2_csf7_lcc \
COLLECTOR_JOB_NAME=collect_charm_seg_cleanup_v2_csf7_lcc \
bash CamCan_Experiment/segmentation_cleanup/submit_charm_segmentation_cleanup.sh
```

Do not mesh the corrected maps unless the collector reports `status=complete`,
`complete=N`, and `incomplete=0`. Use the generated collection manifest as the
input to the existing direct-CHARM mesh and tissue-wall workflow. This step
does not run CHARM segmentation or FEM simulation.
