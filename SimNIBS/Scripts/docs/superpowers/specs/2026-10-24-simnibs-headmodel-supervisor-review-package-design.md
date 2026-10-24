# SimNIBS Head-Model Supervisor Review Package Design

## Purpose

Create one auditable supervisor-review package from a randomly selected member
of the validated nine-subject cohort. The package is a read-only comparison of
the original CHARM, corrected CHARM, original SimNIBS 3.2.6 headreco, and
harmonised SimNIBS 3.2.6 head-model branches. It does not run a simulation or
modify any source dataset.

## Scientific contents

The package contains exactly eight scientific artifacts, all bound to the same
subject:

1. untouched CHARM tissue-label map generated with SimNIBS 4.0.1;
2. corrected CHARM tissue-label map installed in the corrected-v4 scaffold;
3. harmonised tissue-label map retained by the corrected-label SimNIBS 3.2.6
   pipeline;
4. combined tissue-label map exported from the original SimNIBS 3.2.6
   headreco mesh tags onto its `T1fs_conform` grid;
5. subject atlas;
6. original SimNIBS 3.2.6 headreco mesh;
7. SimNIBS 3.2.6 mesh generated from the harmonised map; and
8. corrected-CHARM SimNIBS 4 mesh selected as the median representative mesh
   in the completed left-hippocampus campaign.

The fourth item is explicitly a derived representation of the original
headreco mesh tissue tags. It is not described as a raw CAT12 segmentation.
The sixth and seventh items are distinct native v3 meshes and must be checked
against their separate completion receipts.

## Candidate cohort and random selection

The candidate list is the canonical `BASE9_SUBJECTS` tuple from
`simnibs_326_corrected_label_repeatability.settings`; `sub-CC520083` is not
eligible. Before selection, all nine candidates must have every required source
file and provenance receipt. If any candidate is incomplete, selection fails
closed and the missing evidence is reported.

Selection uses Python's operating-system-backed `secrets.choice` over the
canonical ordered nine-subject list. The selected subject, candidate list,
selection method, timestamp, source bindings, and audit digest are written to
an immutable selection receipt. A rerun reuses the receipt rather than drawing
a new subject.

## Canonical sources

- Original CHARM map and metadata:
  `/mnt/parscratch/users/cop23bi/charm_segmentations/{maps,metadata}`.
- Corrected CHARM map and result receipt:
  `/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds`.
- Original v3 headreco mesh, reference image, and ready marker:
  `/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco`.
- Harmonised map, harmonisation receipt, harmonised v3 mesh, and head-model
  receipt:
  `/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2`.
- Atlas: `/mnt/parscratch/users/cop23bi/ZIPs/atlases`.
- v4 median mesh and selection receipt:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10`.

Receipt-provided paths must resolve inside their expected protected roots and
must name non-empty regular files, never symlinks. Receipt subject IDs, status,
paths, and recorded hashes must agree with the resolved artifact.

## Package layout

The dedicated output root contains the selection receipt, one atomically
promoted package directory, one compressed archive, and a completion receipt.
The package directory contains:

```text
01_original_charm_tissue_labels.nii.gz
02_corrected_charm_tissue_labels.nii.gz
03_harmonized_v3_tissue_labels.nii.gz
04_original_headreco_mesh_tissue_labels.nii.gz
05_subject_atlas.nii.gz
06_v3_original_headreco_mesh.msh
07_v3_harmonized_mesh.msh
08_v4_corrected_charm_median_mesh.msh
README.md
manifest.json
checksums.sha256
```

The manifest records original paths, scientific meaning, producing software,
source and destination hashes, sizes, and provenance receipt paths. The
checksum file uses package-relative paths. The completion receipt binds the
package-directory manifest hash and archive hash.

## Build and validation behavior

The builder re-audits the selected subject immediately before copying. Seven
artifacts are physical, checksum-verified copies. The original headreco label
map is exported using the established SimNIBS 3.2.6 tetrahedral tag assignment
implementation. All work occurs under a temporary directory inside the output
root and is promoted only after all eight artifacts and metadata validate.

Existing complete output is reused only when every checksum and binding still
validates. Partial, unowned, stale, or mismatched output is never overwritten.
No source root is writable and no source artifact is deleted, renamed, or
modified.

## HPC contract

The launcher has `preflight`, `submit`, `status`, and `validate` modes. Submit
requires an exact output-root confirmation, creates or reuses the immutable
selection receipt, and submits one non-array Slurm job. The job uses:

- partition: `sheffield`;
- module: `SimNIBS/3.2.6-foss-2023a`;
- 8 CPUs, 32 GB, 08:00:00;
- no retries and no downstream jobs.

The submission scope is nine audited candidates, one selected subject, one
builder task, and eight expected scientific artifacts. SimNIBS 4 is provenance
for CHARM artifacts, but the build job loads SimNIBS 3.2.6 because only that
version is permitted to export the original headreco mesh-tag volume.

## Exclusions

- no simulations or remeshing;
- no visual renderings or screenshots;
- no inferred clinical comparison;
- no replacement-subject outputs;
- no modification of the completed v3 or v4 campaigns;
- no fallback to a different subject after random selection.
