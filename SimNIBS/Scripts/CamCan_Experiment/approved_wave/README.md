# Approved wave-1 Left Hippocampus campaign

This opt-in workflow starts a fresh experiment for the 89 subjects in
`accepted_wave_1_subjects.txt`, across all ten repeats. It does not read,
modify, archive, or delete the previous CamCan experiment.

The campaign has two dependent stages:

1. A preparation array runs once for each of the 89 unique subjects. It builds
   the complete repeat-01 CHARM `m2m_*` folder, replaces CHARM's generated
   tissue label with the exact supervisor-reviewed flat CHARM label, generates
   repeat 01's experimental mesh, and records the transformed EEG cap.
2. A dependent 890-task array covers every subject-repeat combination. Repeat
   01 reuses its already-completed experimental mesh. Each repeat from 02 to 10
   creates a physical copy of the repeat-01 m2m scaffold, removes the copied
   mesh, runs its own `charm <subject> --mesh`, restores the repeat-01 EEG cap
   byte-for-byte, and then runs the existing FEM simulation and validator.

Consequently, the full campaign retains 890 independently generated meshes
and 890 independently generated simulation outputs. It runs CHARM segmentation
only 89 times, because segmentation is needed only to create the subject
scaffolds. All 890 meshes use the supervisor-reviewed labels. ROAST/custom
segmentation is excluded.

Mesh completion and simulation completion use separate markers. If a FEM task
is requeued after successfully creating its repeat mesh, it reuses that exact
repeat mesh rather than introducing an unplanned additional remesh realization.

The full launcher uses the established Stanage profile:

- SimNIBS `4.0.1-foss-2023a`
- `sheffield`
- 8 CPUs and 32 GB per task
- 8-hour walltime
- preparation array `0-88%50`
- dependent remesh-and-simulate array `0-889%50`
- two self-requeues per task
- `left-hippocampus`
- confirmed `targets.csv` SHA-256
  `97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6`

From `/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts` on Stanage:

```bash
bash CamCan_Experiment/HPC_scripts/submit_approved_wave_mesh_sim.sh
```

The launcher preflights the complete 89-subject preparation scope and complete
890-repeat scope before either `sbatch` call. The second array uses an `afterok`
dependency and therefore cannot start unless all 89 preparation tasks succeed.
