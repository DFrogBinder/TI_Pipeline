#!/bin/bash
#SBATCH --job-name=ti_v3v4_remesh
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${V3_LEFT_HIPPOCAMPUS_CSV:?Set V3_LEFT_HIPPOCAMPUS_CSV.}"
: "${V3_RIGHT_M1_CSV:?Set V3_RIGHT_M1_CSV.}"
: "${V4_LEFT_HIPPOCAMPUS_CSV:?Set V4_LEFT_HIPPOCAMPUS_CSV.}"
: "${V4_RIGHT_M1_CSV:?Set V4_RIGHT_M1_CSV.}"
: "${ANALYSIS_ROOT:?Set ANALYSIS_ROOT.}"

ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$HOME/.conda/envs/ti-post/bin/python}"
ANALYZER="$PIPELINE_DIR/post/compare_simnibs_version_remesh_distributions.py"
RESULTS_DIR="$ANALYSIS_ROOT/results"
MANIFEST="$RESULTS_DIR/analysis_manifest.json"
ARCHIVE="$ANALYSIS_ROOT/simnibs_v3_v4_remesh_distribution_comparison_v1.tar.gz"

if [ ! -x "$ANALYSIS_PYTHON" ]; then
    echo "[ERROR] Analysis Python is missing or not executable: $ANALYSIS_PYTHON" >&2
    exit 2
fi
for required in \
    "$ANALYZER" \
    "$V3_LEFT_HIPPOCAMPUS_CSV" \
    "$V3_RIGHT_M1_CSV" \
    "$V4_LEFT_HIPPOCAMPUS_CSV" \
    "$V4_RIGHT_M1_CSV"; do
    if [ ! -s "$required" ]; then
        echo "[ERROR] Required input is missing or empty: $required" >&2
        exit 2
    fi
done

mkdir -p "$ANALYSIS_ROOT" "$ANALYSIS_ROOT/.mplconfig"
export MPLCONFIGDIR="$ANALYSIS_ROOT/.mplconfig"

if [ -s "$MANIFEST" ]; then
    "$ANALYSIS_PYTHON" - \
        "$MANIFEST" \
        "$V3_LEFT_HIPPOCAMPUS_CSV" \
        "$V3_RIGHT_M1_CSV" \
        "$V4_LEFT_HIPPOCAMPUS_CSV" \
        "$V4_RIGHT_M1_CSV" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_path = Path(sys.argv[1])
inputs = {
    "v3_left_hippocampus_csv": Path(sys.argv[2]),
    "v3_right_m1_csv": Path(sys.argv[3]),
    "v4_left_hippocampus_csv": Path(sys.argv[4]),
    "v4_right_m1_csv": Path(sys.argv[5]),
}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "input_rows": 3200,
    "selected_remesh_rows": 1600,
    "run_percentile_rows": 2400,
    "participant_target_primary_rows": 20,
    "participant_inference_rows": 6,
    "inferential_unit": "participant",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"Existing manifest has unexpected {key}: {manifest.get(key)!r}")
if manifest.get("technical_repeats_are_inferential_units") is not False:
    raise SystemExit("Existing manifest has the wrong inferential-unit declaration")
for key, path in inputs.items():
    if manifest.get("inputs", {}).get(key, {}).get("sha256") != sha256(path):
        raise SystemExit(f"Existing analysis input hash differs for {key}")
for artifact in manifest.get("artifacts", []):
    path = manifest_path.parent / artifact["path"]
    if not path.is_file():
        raise SystemExit(f"Existing artifact is missing: {path}")
    if int(artifact["size_bytes"]) != path.stat().st_size:
        raise SystemExit(f"Existing artifact size differs: {path}")
    if artifact["sha256"] != sha256(path):
        raise SystemExit(f"Existing artifact hash differs: {path}")
print("[INFO] Existing complete analysis matches all four immutable inputs.")
PY
else
    if [ -e "$RESULTS_DIR" ]; then
        echo "[ERROR] Refusing to merge with incomplete results: $RESULTS_DIR" >&2
        exit 2
    fi
    ATTEMPT_DIR="$ANALYSIS_ROOT/.results-${SLURM_JOB_ID:-manual}"
    if [ -e "$ATTEMPT_DIR" ]; then
        echo "[ERROR] Attempt directory already exists: $ATTEMPT_DIR" >&2
        exit 2
    fi

    "$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas'
    "$ANALYSIS_PYTHON" -u "$ANALYZER" run \
        --v3-left-hippocampus-csv "$V3_LEFT_HIPPOCAMPUS_CSV" \
        --v3-right-m1-csv "$V3_RIGHT_M1_CSV" \
        --v4-left-hippocampus-csv "$V4_LEFT_HIPPOCAMPUS_CSV" \
        --v4-right-m1-csv "$V4_RIGHT_M1_CSV" \
        --out-dir "$ATTEMPT_DIR"

    "$ANALYSIS_PYTHON" - "$ATTEMPT_DIR/analysis_manifest.json" <<'PY'
import json
import sys
from pathlib import Path


manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "input_rows": 3200,
    "selected_remesh_rows": 1600,
    "run_percentile_rows": 2400,
    "participant_metric_rows": 60,
    "participant_target_primary_rows": 20,
    "participant_inference_rows": 6,
    "participants": 10,
    "inferential_unit": "participant",
    "meshing_or_fem_tasks": 0,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"Analysis manifest has unexpected {key}: {manifest.get(key)!r}")
if manifest.get("technical_repeats_are_inferential_units") is not False:
    raise SystemExit("Analysis incorrectly marks technical repeats as inferential units")
if set(manifest.get("targets", [])) != {"left_hippocampus", "right_m1"}:
    raise SystemExit("Analysis manifest has the wrong targets")
if set(manifest.get("versions", [])) != {"v3", "v4"}:
    raise SystemExit("Analysis manifest has the wrong versions")
PY
    mv "$ATTEMPT_DIR" "$RESULTS_DIR"
fi

TEMP_ARCHIVE="$ANALYSIS_ROOT/.simnibs_v3_v4_remesh_distribution_comparison_v1.${SLURM_JOB_ID:-manual}.tmp.tar.gz"
tar -C "$RESULTS_DIR" -czf "$TEMP_ARCHIVE" .
mv "$TEMP_ARCHIVE" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

printf 'job_id\tstatus\tmanifest\tarchive\tsource_policy\tmeshing_or_fem_tasks\n%s\tcomplete\t%s\t%s\tread-only\t0\n' \
    "${SLURM_JOB_ID:-manual}" "$MANIFEST" "$ARCHIVE" \
    > "$ANALYSIS_ROOT/completion-${SLURM_JOB_ID:-manual}.receipt.tsv"

echo "[INFO] Analysis manifest: $MANIFEST"
echo "[INFO] Download archive:  $ARCHIVE"
echo "[INFO] Source policy:     read-only"
echo "[INFO] Meshing/FEM tasks: 0"
