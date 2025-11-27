Resource monitoring for a running Slurm job
===========================================

- Live from login node:
  - `sstat -j <jobid>.batch --format=JobID,AveCPU,AveRSS,MaxRSS,AveDiskRead,AveDiskWrite`
  - `sacct -j <jobid> -o JobID,AllocCPUS,Elapsed,TotalCPU,MaxRSS,State` (works during/after)
  - `seff <jobid>` (after job finishes) for efficiency summary.

- From inside the allocation (if allowed): `srun --overlap --jobid=<jobid> --pty bash` then:
  - `htop` or `top`
  - `ps -eo pid,psr,pcpu,pmem,cmd --sort=-pcpu | head`
  - `free -h`

- Optional snapshot in your batch script:
  ```bash
  echo "Resource snapshot:"
  date
  ps -eo pid,psr,pcpu,pmem,cmd --sort=-pcpu | head
  free -h
  ```

- Notes:
  - Use the `.batch` step suffix with `sstat` while running.
  - `sacct`/`seff` are best after the job ends for peak RSS and CPU efficiency.
