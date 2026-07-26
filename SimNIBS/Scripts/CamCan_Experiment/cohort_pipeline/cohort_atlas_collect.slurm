#!/bin/bash
#SBATCH --job-name=cohort_atlas_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

set -euo pipefail

: "${TI_COHORT_ATLAS_SUBJECTS_FILE:?Missing TI_COHORT_ATLAS_SUBJECTS_FILE.}"
: "${TI_COHORT_ATLAS_FLAT_ROOT:?Missing TI_COHORT_ATLAS_FLAT_ROOT.}"
: "${TI_COHORT_ATLAS_VALIDATION_TSV:?Missing TI_COHORT_ATLAS_VALIDATION_TSV.}"
: "${TI_COHORT_ATLAS_VALIDATION_JSON:?Missing TI_COHORT_ATLAS_VALIDATION_JSON.}"

mkdir -p "$(dirname "${TI_COHORT_ATLAS_VALIDATION_TSV}")"
TMP_TSV="${TI_COHORT_ATLAS_VALIDATION_TSV}.tmp.${SLURM_JOB_ID}"
printf '%s\n' $'subject\tstatus\tatlas_path\tbytes\tmessage' > "${TMP_TSV}"

TOTAL=0
COMPLETE=0
INCOMPLETE=0
while read -r subject; do
    [ -n "${subject}" ] || continue
    TOTAL=$((TOTAL + 1))
    atlas_path=""
    for candidate in \
        "${TI_COHORT_ATLAS_FLAT_ROOT}/${subject}.nii.gz" \
        "${TI_COHORT_ATLAS_FLAT_ROOT}/${subject}.nii"
    do
        if [ -s "${candidate}" ]; then
            atlas_path="${candidate}"
            break
        fi
    done
    if [ -n "${atlas_path}" ]; then
        bytes="$(stat -c '%s' "${atlas_path}")"
        printf '%s\tcomplete\t%s\t%s\t\n' \
            "${subject}" "${atlas_path}" "${bytes}" >> "${TMP_TSV}"
        COMPLETE=$((COMPLETE + 1))
    else
        printf '%s\tincomplete\t\t0\tmissing subject-space Destrieux atlas\n' \
            "${subject}" >> "${TMP_TSV}"
        INCOMPLETE=$((INCOMPLETE + 1))
    fi
done < "${TI_COHORT_ATLAS_SUBJECTS_FILE}"
mv "${TMP_TSV}" "${TI_COHORT_ATLAS_VALIDATION_TSV}"

STATUS="complete"
if [ "${INCOMPLETE}" -ne 0 ]; then
    STATUS="incomplete"
fi
python3 -c \
    'import json,sys; json.dump({"status":sys.argv[1],"subjects":int(sys.argv[2]),"complete":int(sys.argv[3]),"incomplete":int(sys.argv[4]),"atlas":"FreeSurfer Destrieux aparc.a2009s+aseg","validation":sys.argv[5]},open(sys.argv[6],"w"),indent=2,sort_keys=True); print(json.dumps({"status":sys.argv[1],"subjects":int(sys.argv[2]),"complete":int(sys.argv[3]),"incomplete":int(sys.argv[4])},indent=2))' \
    "${STATUS}" \
    "${TOTAL}" \
    "${COMPLETE}" \
    "${INCOMPLETE}" \
    "${TI_COHORT_ATLAS_VALIDATION_TSV}" \
    "${TI_COHORT_ATLAS_VALIDATION_JSON}"

if [ "${INCOMPLETE}" -ne 0 ]; then
    echo "[ERROR] ${INCOMPLETE}/${TOTAL} subject-space atlases remain missing." >&2
    exit 1
fi
echo "[INFO] All ${TOTAL} subject-space Destrieux atlases are present."
echo "[INFO] The cohort post-processing preflight may now be rerun."
