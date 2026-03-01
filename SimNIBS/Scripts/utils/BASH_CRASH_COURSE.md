# Bash Crash Course for This Folder

This is just enough Bash to read and modify:
- `register_seg_to_anat_fsl.sh`
- `register_seg_to_anat_fsl_batch.sh`

## 1) Script skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail
```

- `set -e`: stop on command failure.
- `set -u`: fail on undefined variable.
- `pipefail`: pipeline fails if any stage fails.

These scripts are fail-fast by design.

## 2) Arguments and defaults

```bash
if [[ $# -lt 3 || $# -gt 4 ]]; then ... fi
SEG="$1"
DOF="${4:-6}"
ROOT_DIR="$1"; shift
```

- `$#` = argument count.
- `$1`, `$2`... = positional arguments.
- `${4:-6}` = default value if arg 4 is missing.
- `shift` drops the first arg so the rest can be parsed.

## 3) Conditionals and tests

```bash
[[ -f "$file" ]]      # file exists
[[ -d "$dir" ]]       # directory exists
[[ -z "$value" ]]     # empty string
[[ "$JOBS" =~ ^[0-9]+$ ]]  # regex check
```

Used with `if ...; then ... else ... fi`.

## 4) Functions

```bash
usage() { ... }
find_nifti() { ... }
wait_for_one_job() { ... }
```

- `local var=...` keeps function variables local.
- `return 0` = success, non-zero = failure.

## 5) Loop patterns used here

```bash
for anat_dir in "${anat_dirs[@]}"; do ... done
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs) ... ;;
    *) ... ;;
  esac
done
```

- `"${array[@]}"` safely iterates array items.
- `while + case` is a standard flag parser.

## 6) Command substitution and path handling

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
subject_id="$(basename "$(dirname "$anat_dir")")"
```

- `$(...)` captures command output.
- Used to derive script-relative and subject-relative paths.

## 7) Exit codes and dependency checks

```bash
command -v flirt >/dev/null 2>&1 || { echo "Error"; exit 1; }
```

- Most commands: `0` success, non-zero failure.
- `||` runs the right side if left side fails.

## 8) Redirection and logging

```bash
... >"$log_file" 2>&1
```

- `>` writes stdout to file.
- `2>&1` merges stderr into stdout.

Batch mode stores one log per subject, then prints failed tails.

## 9) Arrays and globs

```bash
shopt -s nullglob
anat_dirs=( "$ROOT_DIR"/*/anat )
status_files+=( "$status_file" )
```

- `nullglob`: unmatched glob expands to empty list (not literal `*`).
- `arr=(...)` create array, `arr+=(...)` append.

## 10) Parallel execution model in batch script

```bash
( run_subject ) &
pid="$!"
wait -n
```

- `( ... ) &` starts a background subshell.
- `$!` is last background PID.
- `wait -n` waits for one job to complete.

Script logic:
1. Discover subject `anat` folders.
2. Validate required input files per subject.
3. Launch up to `-j/--jobs` registrations in parallel.
4. Collect per-subject status files and summarize.

## 11) Cleanup

```bash
tmp_dir="$(mktemp -d ...)"
trap 'rm -rf "$tmp_dir"' EXIT
```

Temporary status/log directory is always removed on exit.

## 12) End-to-end flow

1. `register_seg_to_anat_fsl_batch.sh` loops subjects and dispatches work.
2. `register_seg_to_anat_fsl.sh` runs `flirt` and `convert_xfm`.
3. Outputs are `<out>_seg_in_anat.nii.gz`, `<out>_seg2anat.mat`, `<out>_anat2seg.mat`.

## 13) Useful commands

```bash
bash -n register_seg_to_anat_fsl.sh
bash -n register_seg_to_anat_fsl_batch.sh

bash register_seg_to_anat_fsl_batch.sh /data/study 6
bash register_seg_to_anat_fsl_batch.sh /data/study 6 -j 4
```
