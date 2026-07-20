# Approved wave-1 Left Hippocampus campaign

This opt-in workflow starts a fresh experiment for the 89 subjects in
`accepted_wave_1_subjects.txt`, across all ten repeats. It does not read,
modify, archive, or delete the previous 175-subject CamCan experiment.

Each of the 890 independent array tasks:

1. verifies its T1, T2, and QC-approved CHARM label against preflight hashes;
2. creates a fresh repeat workspace;
3. runs the CHARM registration/segmentation prerequisites needed to reconstruct
   the simulation support files, without requesting surfaces or a mesh;
4. replaces the newly generated label with the exact approved flat label;
5. runs `charm <subject> --mesh` and verifies the approved label is unchanged;
6. verifies the optimized electrodes exist in the transformed EEG cap;
7. runs the existing `TI_runner_multi-core.py --reuse-existing-mesh` path with
   the confirmed Left Hippocampus montage; and
8. runs the existing simulation output validator before marking the task done.

Mesh and simulation completion have separate JSON markers. A retry after a
simulation failure reuses the completed mesh rather than generating a new
repeat mesh.

The full launcher uses the established Stanage profile:

- SimNIBS `4.0.1-foss-2023a`
- `sheffield`
- 8 CPUs and 32 GB per task
- 8-hour walltime
- `0-889%50`
- two self-requeues
- `left-hippocampus`
- confirmed `targets.csv` SHA-256
  `97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6`

From `/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts` on Stanage:

```bash
bash CamCan_Experiment/HPC_scripts/submit_approved_wave_mesh_sim.sh
```

The launcher runs the complete 89-subject/890-task preflight before calling
`sbatch`. Any missing or ambiguous MRI input, missing approved map, hash error,
scope mismatch, montage mismatch, or incompatible Slurm array limit blocks the
submission.

