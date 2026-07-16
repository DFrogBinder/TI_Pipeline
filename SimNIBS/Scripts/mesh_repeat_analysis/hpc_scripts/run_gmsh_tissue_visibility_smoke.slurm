#!/bin/bash
#SBATCH --job-name=gmsh_tissue_smoke
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=gmsh_tissue_smoke_%j.out
#SBATCH --error=gmsh_tissue_smoke_%j.err

set -euo pipefail

MESH="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs/Left_Hippocampus_Data_01/sub-CC110056/anat/m2m_sub-CC110056/sub-CC110056.msh"
OUT_ROOT="/mnt/parscratch/users/cop23bi/mesh-wall/gmsh_tissue_visibility_smoke"
GMSH_MODULE="gmsh/4.11.1-foss-2022b"
XVFB_MODULE="Xvfb/21.1.6-GCCcore-12.2.0"
TISSUE_TAG="5"

module purge
module use "$HOME/modules"
module load "${GMSH_MODULE}"
module load "${XVFB_MODULE}"

if [ ! -f "${MESH}" ]; then
    echo "[ERROR] Smoke-test mesh does not exist: ${MESH}" >&2
    exit 2
fi
if ! command -v gmsh >/dev/null 2>&1 || ! gmsh -version >/dev/null 2>&1; then
    echo "[ERROR] The configured Gmsh module did not provide a working executable." >&2
    exit 2
fi
if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[ERROR] The configured Xvfb module did not provide xvfb-run." >&2
    exit 2
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_ROOT}/${JOB_ID}"
GEO_FILE="${OUT_DIR}/scalp_front_back.geo"
FRONT_PNG="${OUT_DIR}/scalp_front.png"
BACK_PNG="${OUT_DIR}/scalp_back.png"
mkdir -p "${OUT_DIR}"

cat > "${GEO_FILE}" <<EOF
Merge "${MESH}";
General.Terminal = 1;
General.SmallAxes = 0;
General.Axes = 0;
General.Orthographic = 1;
General.Trackball = 0;
General.RotationCenterGravity = 1;
General.GraphicsWidth = 600;
General.GraphicsHeight = 600;
Mesh.Points = 0;
Mesh.Lines = 0;
Mesh.SurfaceEdges = 0;
Mesh.SurfaceFaces = 0;
Mesh.VolumeEdges = 0;
Mesh.VolumeFaces = 1;
Mesh.Light = 1;
Mesh.LightLines = 1;
Mesh.LightTwoSide = 1;
Print.Background = 0;
Print.Width = 600;
Print.Height = 600;

Hide {:}
Recursive Show { Physical Volume{${TISSUE_TAG}}; }
General.RotationX = 0;
General.RotationY = 0;
General.RotationZ = 0;
Draw;
Print "${FRONT_PNG}";

General.RotationZ = 180;
Draw;
Print "${BACK_PNG}";
Exit;
EOF

echo "[INFO] Direct Gmsh tissue-visibility smoke test"
echo "[INFO] Job ID:       ${JOB_ID}"
echo "[INFO] Host:         $(hostname)"
echo "[INFO] Mesh:         ${MESH}"
echo "[INFO] Tissue tag:   ${TISSUE_TAG} (scalp)"
echo "[INFO] Gmsh:         $(command -v gmsh) ($(gmsh -version 2>&1))"
echo "[INFO] Output:       ${OUT_DIR}"

START_SECONDS="${SECONDS}"
if xvfb-run -a gmsh "${GEO_FILE}" -nopopup -v 4; then
    :
else
    status="$?"
    echo "[ERROR] Gmsh visibility render failed (status=${status})." >&2
    exit 2
fi

for png in "${FRONT_PNG}" "${BACK_PNG}"; do
    if [ ! -s "${png}" ]; then
        echo "[ERROR] Gmsh did not produce a non-empty image: ${png}" >&2
        exit 2
    fi
done

echo "[INFO] PASS: one mesh load produced front and back scalp images in $((SECONDS - START_SECONDS)) seconds"
ls -lh "${FRONT_PNG}" "${BACK_PNG}"
