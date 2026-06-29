import csv

import numpy as np

from mesh_repeat_analysis.post.mesh_qc import run_mesh_qc
from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays


def test_cli_writes_csv_reports_in_skip_render_mode(tmp_path, monkeypatch):
    mesh_path = tmp_path / "Pallidum" / "sub-CC110056" / "repeat_01" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    def fake_loader(path):
        assert path == mesh_path
        return SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        )

    monkeypatch.setattr(run_mesh_qc, "load_surface_arrays", fake_loader)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--mesh-glob",
            "*.msh",
            "--skip-renders",
        ]
    )

    assert rc == 0
    with (out_dir / "qc_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["roi"] == "Pallidum"
    assert rows[0]["subject"] == "sub-CC110056"
    assert rows[0]["status"] == "OK"

