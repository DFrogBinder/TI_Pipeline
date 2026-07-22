# CHARM segmentation cleanup for second-round QC

This workflow implements the scripted corrections requested after the second
round of tissue-wall review. It operates on collected flat CHARM label maps
and never modifies them in place.

## Scientific status

The corrected CSF and skin operations reproduce the supplied Seg3D intent:

- close the cumulative WM + GM + CSF envelope by 5 voxels;
- optionally open that closed envelope with a smaller Euclidean ball to remove
  narrow outward bumps that closing alone cannot remove;
- close the cumulative whole-head tissue envelope by 10 voxels; and
- rebuild the exclusive CHARM label map with the supplied tissue hierarchy.

Blood is deliberately excluded from the CSF mask before closing and restored
as label 9 immediately after CSF during reconstruction. Including blood in the
working CSF mask caused dilation around vessels to leave a label-3 halo after
the vessel core was restored. `--include-blood-in-csf` exists only to reproduce
the invalid legacy v1/v2 candidates and must not be used for corrected QC data.

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

An opt-in candidate can also filter the cumulative
WM + GM + CSF envelope to its largest 26-connected component before
closing it. This is enabled with `--csf-component-policy largest` (or the
matching launcher environment variable). The component-filter default remains
`none`. Filtering the cumulative envelope removes
fully disconnected islands; it does not remove bumps that remain connected to
the main envelope. Increasing the closing radius fills larger grooves and
concavities but is not equivalent to erosion-based removal of outward bumps.
The opt-in `--csf-opening-radius` performs erosion followed by dilation after
closing. The v4 tuning candidate uses closing radius 7 and opening radius 3.
When the CSF component policy is `largest`, v4 enforces it both before closing
and again after opening because opening can sever thin bridges and create tiny
detached fragments.
Opening can remove genuine narrow CSF anatomy, so this candidate must be
reviewed as isolated before/after CSF masks before cohort-wide use.

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
/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v3_no_blood/
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
    --output-root /path/to/corrected_v3_no_blood \
    --manifest /path/to/corrected_v3_no_blood/campaign/cleanup_manifest.tsv \
    --summary /path/to/corrected_v3_no_blood/campaign/preflight.json \
    --expected-subjects EXPECTED_COUNT
```

Then process the full manifest with a memory-conscious worker count:

```bash
python3 CamCan_Experiment/segmentation_cleanup/workflow.py run-all \
    --manifest /path/to/corrected_v3_no_blood/campaign/cleanup_manifest.tsv \
    --workers 4
```

Validation creates the collection manifest consumed directly by
`mesh_collected_segmentations.py`:

```bash
python3 CamCan_Experiment/segmentation_cleanup/workflow.py validate \
    --manifest /path/to/corrected_v3_no_blood/campaign/cleanup_manifest.tsv \
    --validation /path/to/corrected_v3_no_blood/campaign/validation.tsv \
    --summary /path/to/corrected_v3_no_blood/campaign/validation.json \
    --collection-manifest /path/to/corrected_v3_no_blood/collection/charm_segmentation_manifest.tsv \
    --checksums /path/to/corrected_v3_no_blood/collection/sha256sums.txt
```

For a deliberately bounded local visual check, the comparison utility applies
the complete correction but writes only binary before/after CSF masks and a
signed difference image. It enforces the stated subject count and never edits
the source maps:

```bash
python3 CamCan_Experiment/segmentation_cleanup/compare_csf_smoothing.py \
    --maps-root /path/to/original/maps \
    --output-root /path/to/csf_smoothing_comparison \
    --subject sub-CC110056 \
    --subject sub-CC412021 \
    --subject sub-CC723197 \
    --expected-subjects 3 \
    --csf-closing-radius 7 \
    --csf-opening-radius 3
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

The default remains one subject process per array element. An opt-in packed
mode sets `TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK=2`. Each element then owns a
two-row manifest shard and launches two single-threaded Python processes. For
652 subjects this produces 326 array elements; `%50` permits at most 100
simultaneous subjects. A full-resolution v4 worker measured 7.6 GB peak RSS,
so the two-worker profile uses 24 GB rather than the unsafe 16 GB allocation.
If one worker fails, the shard requeues; a completed companion result is
recognized and skipped on the retry.
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

The v2 CSF-radius-7 candidate included blood in the CSF working mask and is
scientifically invalid. For the corrected v3 CSF-radius-7/largest-component
candidate, use a fresh output root and keep blood exclusion explicit:

```bash
OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v3_csf7_lcc_no_blood \
TI_CHARM_CLEANUP_CSF_RADIUS=7 \
TI_CHARM_CLEANUP_CSF_INCLUDE_BLOOD=0 \
TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY=largest \
TIME_LIMIT=08:00:00 \
COLLECTOR_MEMORY=16G \
COLLECTOR_TIME=08:00:00 \
JOB_NAME=charm_seg_cleanup_v3_csf7_lcc_no_blood \
COLLECTOR_JOB_NAME=collect_charm_seg_cleanup_v3_csf7_lcc_no_blood \
bash CamCan_Experiment/segmentation_cleanup/submit_charm_segmentation_cleanup.sh
```

Do not launch v4 cohort processing until its three-subject CSF masks have been
reviewed. Once accepted, v4 adds this explicit setting to the v3 profile and
uses another fresh output root:

```bash
OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v4_csf7_open3_lcc_no_blood \
TI_CHARM_CLEANUP_CSF_RADIUS=7 \
TI_CHARM_CLEANUP_CSF_OPENING_RADIUS=3 \
TI_CHARM_CLEANUP_CSF_INCLUDE_BLOOD=0 \
TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY=largest \
TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK=2 \
MAX_CONCURRENT_TASKS=50 \
CPUS_PER_TASK=2 \
MEMORY=24G \
TIME_LIMIT=08:00:00 \
COLLECTOR_MEMORY=16G \
COLLECTOR_TIME=08:00:00 \
JOB_NAME=charm_seg_cleanup_v4_csf7_open3_lcc_no_blood \
COLLECTOR_JOB_NAME=collect_charm_seg_cleanup_v4_csf7_open3_lcc_no_blood \
bash CamCan_Experiment/segmentation_cleanup/submit_charm_segmentation_cleanup.sh
```

Do not mesh the corrected maps unless the collector reports `status=complete`,
`complete=N`, and `incomplete=0`. Use the generated collection manifest as the
input to the existing direct-CHARM mesh and tissue-wall workflow. This step
does not run CHARM segmentation or FEM simulation.
