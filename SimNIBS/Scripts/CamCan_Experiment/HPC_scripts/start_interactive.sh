#!/bin/bash
set -euo pipefail

# Launch an interactive Slurm shell directly on a worker node.
# Defaults mirror the post-processing batch wrapper and can be overridden
# with environment variables or extra srun flags passed to this script.

PARTITION="${SLURM_INTERACTIVE_PARTITION:-sheffield}"
NODES="${SLURM_INTERACTIVE_NODES:-1}"
NTASKS="${SLURM_INTERACTIVE_NTASKS:-1}"
CPUS_PER_TASK="${SLURM_INTERACTIVE_CPUS_PER_TASK:-12}"
MEM="${SLURM_INTERACTIVE_MEM:-24G}"
TIME_LIMIT="${SLURM_INTERACTIVE_TIME:-08:00:00}"
JOB_NAME="${SLURM_INTERACTIVE_JOB_NAME:-ti_post_interactive}"
ACCOUNT="${SLURM_INTERACTIVE_ACCOUNT:-}"
QOS="${SLURM_INTERACTIVE_QOS:-}"
SHELL_BIN="${SLURM_INTERACTIVE_SHELL:-/bin/bash}"

if ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] srun was not found in PATH." >&2
    echo "[ERROR] Load your Slurm environment on the login node first." >&2
    exit 2
fi

if [[ "${SHELL_BIN}" == */* ]]; then
    SHELL_CMD="${SHELL_BIN}"
else
    SHELL_CMD="$(command -v "${SHELL_BIN}" || true)"
fi

if [[ -z "${SHELL_CMD}" || ! -x "${SHELL_CMD}" ]]; then
    echo "[ERROR] Interactive shell not found or not executable: ${SHELL_BIN}" >&2
    exit 2
fi

SRUN_ARGS=(--pty --export=ALL)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[INFO] Existing Slurm allocation detected: ${SLURM_JOB_ID}"
    echo "[INFO] Opening an interactive shell inside the current allocation."
    exec srun "${SRUN_ARGS[@]}" --overlap "$@" "${SHELL_CMD}" -l
fi

SRUN_ARGS+=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --nodes="${NODES}"
    --ntasks="${NTASKS}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --time="${TIME_LIMIT}"
)

if [[ -n "${ACCOUNT}" ]]; then
    SRUN_ARGS+=(--account="${ACCOUNT}")
fi

if [[ -n "${QOS}" ]]; then
    SRUN_ARGS+=(--qos="${QOS}")
fi

echo "[INFO] Requesting interactive worker shell with:"
echo "[INFO]   partition=${PARTITION} nodes=${NODES} ntasks=${NTASKS} cpus=${CPUS_PER_TASK} mem=${MEM} time=${TIME_LIMIT}"
echo "[INFO]   job_name=${JOB_NAME} shell=${SHELL_CMD}"
if [[ $# -gt 0 ]]; then
    echo "[INFO]   extra srun args: $*"
fi

exec srun "${SRUN_ARGS[@]}" "$@" "${SHELL_CMD}" -l
