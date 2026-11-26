# Historical Component-Field Analysis HPC Implementation Plan

1. Add fail-closed consolidated-dataset discovery and task planning for the
   exact 3,200 historical simulations.
2. Add a historical native extractor that reuses the v2 ROI and precision
   primitives while validating stimulation provenance and tag 2101.
3. Add resumable per-task receipts and exact version collectors.
4. Add a separate preliminary 3.2.6/4.0.1 analysis validator and workflow,
   leaving the final 3.2.6/4.6 workflow unchanged.
5. Add a dedicated CLI and guarded Slurm launchers whose default is preflight
   and whose submission path requires `--submit`.
6. Add unit and workflow tests first, then implement until the focused suite,
   shell syntax checks, compile checks, and diff checks pass.
7. Self-review the final diff, commit the isolated branch, and report the one
   user-run HPC preflight command needed to confirm live paths.
