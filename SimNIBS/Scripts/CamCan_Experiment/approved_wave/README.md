# Approved wave-1 Left Hippocampus campaign

This opt-in workflow starts a fresh experiment for the 89 subjects in
`accepted_wave_1_subjects.txt`, across all ten repeats. It does not read,
modify, archive, or delete the previous CamCan experiment.

The campaign has two dependent stages:

1. A preparation array runs exactly once for each of the 89 unique subjects.
   It creates the complete repeat-01 CHARM `m2m_*` support tree, immediately
   replaces CHARM's generated tissue label with the exact supervisor-reviewed
   flat CHARM label, creates one mesh from that reviewed label, and verifies
   the mesh, label hash, and optimized EEG-cap positions.
2. A dependent simulation array contains all 890 subject-repeat combinations.
   Repeat 01 uses the canonical support tree directly. Repeats 02-10 receive
   relative links to the canonical T1, T2, and complete `m2m_*` tree, avoiding
   nine redundant copies of each large mesh. Every task runs the existing
   `TI_runner_multi-core.py --reuse-existing-mesh` path and validates its own
   repeat-specific output.

The simulation stage never invokes CHARM segmentation or meshing. The shared
canonical `m2m_*` tree is read-only during simulations. ROAST/custom
segmentation is excluded by both the workflow and the existing reuse runner.

The full launcher uses the established Stanage profile:

- SimNIBS `4.0.1-foss-2023a`
- `sheffield`
- 8 CPUs and 32 GB per task
- 8-hour walltime
- preparation array `0-88%50`
- dependent simulation array `0-889%50`
- two self-requeues per task
- `left-hippocampus`
- confirmed `targets.csv` SHA-256
  `97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6`

From `/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts` on Stanage:

```bash
bash CamCan_Experiment/HPC_scripts/submit_approved_wave_mesh_sim.sh
```

The launcher preflights the complete 89-subject preparation scope and the
complete 890-simulation scope before either `sbatch` call. The simulation array
uses an `afterok` dependency and therefore cannot start unless every subject
preparation task succeeds. If the second submission itself fails, the launcher
cancels the first submission to avoid leaving an unintended partial campaign.
