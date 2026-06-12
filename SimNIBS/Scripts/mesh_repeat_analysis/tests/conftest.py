import sys
from pathlib import Path


MESH_REPEAT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = MESH_REPEAT_ROOT.parents[0]

for path in (MESH_REPEAT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
