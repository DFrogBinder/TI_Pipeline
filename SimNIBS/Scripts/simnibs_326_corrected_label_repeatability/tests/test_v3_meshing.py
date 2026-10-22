from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simnibs_326_corrected_label_repeatability.v3_meshing import (
    MeshValidationError,
    MeshingAPIError,
    V3MeshingAPI,
    build_surfaces,
    build_volume_mesh,
    probe_v3_meshing_api,
    validate_mesh,
)


class FakeMesh:
    def fix_surface_labels(self) -> None:
        pass

    def fix_thin_tetrahedra(self) -> None:
        pass

    def fix_th_node_ordering(self) -> None:
        pass

    def fix_tr_node_ordering(self) -> None:
        pass

    def elements_volumes_and_areas(self):
        return SimpleNamespace(value=np.ones(12))


def make_surface_meshes(
    input_files,
    out_dir,
    temp_dir,
    vertex_density=0.5,
    catsurf=False,
):
    del input_files, out_dir, temp_dir, vertex_density, catsurf
    return [], []


def make_volume_mesh(subject_id, input_files, out_dir, keep_air):
    del subject_id, input_files, out_dir, keep_air
    return "raw.msh"


def relabel_compartments(mesh, old, new):
    del old, new
    return mesh


def volume_mesh_relabel(mesh, surfaces, region_number):
    del surfaces, region_number
    return mesh


def read_msh(filename, m=None, buffered=False, skip_data=False):
    del filename, m, buffered, skip_data
    return FakeMesh()


def write_msh(mesh, filename=None, mode="binary"):
    del mesh, filename, mode


def _api() -> V3MeshingAPI:
    simnibs = SimpleNamespace(__version__="3.2.6", __file__="/fake/simnibs/__init__.py")
    headreco = SimpleNamespace(
        make_surface_meshes=make_surface_meshes,
        make_volume_mesh=make_volume_mesh,
        __file__="/fake/simnibs/msh/headreco.py",
    )
    hmutils = SimpleNamespace(
        relabel_compartments=relabel_compartments,
        volume_mesh_relabel=volume_mesh_relabel,
        path2bin=lambda name: f"/fake/bin/{name}",
        __file__="/fake/simnibs/msh/hmutils.py",
    )
    mesh_io = SimpleNamespace(
        read_msh=read_msh,
        write_msh=write_msh,
        Msh=FakeMesh,
        __file__="/fake/simnibs/msh/mesh_io.py",
    )
    return V3MeshingAPI(
        simnibs=simnibs,
        headreco=headreco,
        hmutils=hmutils,
        mesh_io=mesh_io,
    )


def test_probe_accepts_exact_v326_api_contract() -> None:
    report = probe_v3_meshing_api(_api())

    assert report["status"] == "ready"
    assert report["simnibs_version"] == "3.2.6"
    assert report["module_files"] == {
        "simnibs": "/fake/simnibs/__init__.py",
        "headreco": "/fake/simnibs/msh/headreco.py",
        "hmutils": "/fake/simnibs/msh/hmutils.py",
        "mesh_io": "/fake/simnibs/msh/mesh_io.py",
    }
    assert report["binaries"] == {
        "gmsh": "/fake/bin/gmsh",
        "meshfix": "/fake/bin/meshfix",
    }
    assert report["signatures"]["make_surface_meshes"] == (
        "(input_files, out_dir, temp_dir, vertex_density=0.5, catsurf=False)"
    )
    assert report["signatures"]["make_volume_mesh"] == (
        "(subject_id, input_files, out_dir, keep_air)"
    )
    assert report["mesh_cleanup_methods"] == [
        "fix_surface_labels",
        "fix_thin_tetrahedra",
        "fix_th_node_ordering",
        "fix_tr_node_ordering",
        "elements_volumes_and_areas",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("version", "version"),
        ("surface_missing", "make_surface_meshes"),
        ("surface_signature", "make_surface_meshes"),
        ("volume_missing", "make_volume_mesh"),
        ("volume_signature", "make_volume_mesh"),
        ("relabel_missing", "relabel_compartments"),
        ("volume_relabel_missing", "volume_mesh_relabel"),
        ("read_missing", "read_msh"),
        ("read_no_buffered", "buffered"),
        ("write_missing", "write_msh"),
        ("cleanup_missing", "fix_thin_tetrahedra"),
        ("binary_missing", "gmsh"),
    ],
)
def test_probe_fails_closed_for_api_drift(mutation: str, message: str) -> None:
    api = _api()
    if mutation == "version":
        api.simnibs.__version__ = "4.0.1"
    elif mutation == "surface_missing":
        api.headreco.make_surface_meshes = None
    elif mutation == "surface_signature":
        api.headreco.make_surface_meshes = lambda input_files, out_dir: None
    elif mutation == "volume_missing":
        api.headreco.make_volume_mesh = None
    elif mutation == "volume_signature":
        api.headreco.make_volume_mesh = lambda subject_id, input_files: None
    elif mutation == "relabel_missing":
        api.hmutils.relabel_compartments = None
    elif mutation == "volume_relabel_missing":
        api.hmutils.volume_mesh_relabel = None
    elif mutation == "read_missing":
        api.mesh_io.read_msh = None
    elif mutation == "read_no_buffered":
        api.mesh_io.read_msh = lambda filename: FakeMesh()
    elif mutation == "write_missing":
        api.mesh_io.write_msh = None
    elif mutation == "cleanup_missing":
        class IncompleteMesh(FakeMesh):
            fix_thin_tetrahedra = None

        api.mesh_io.Msh = IncompleteMesh
    elif mutation == "binary_missing":
        api.hmutils.path2bin = lambda name: "" if name == "gmsh" else f"/fake/{name}"
    else:  # pragma: no cover - guards the test table
        raise AssertionError(mutation)

    with pytest.raises(MeshingAPIError, match=message):
        probe_v3_meshing_api(api)


def test_surface_generation_uses_sorted_masks_and_pinned_options(tmp_path: Path) -> None:
    events: list[tuple] = []
    api = _recording_api(tmp_path, events)
    masks = {}
    for name in ("WM", "SKIN", "CSF", "AIR", "GM", "EYES", "BONE"):
        path = tmp_path / "masks" / f"MASK_{name}.nii.gz"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        masks[name] = path
    m2m = tmp_path / "m2m"
    scratch = tmp_path / "scratch"

    result = build_surfaces(api, masks, m2m, scratch)

    event = next(item for item in events if item[0] == "make_surface_meshes")
    assert event[1] == tuple(
        str(path) for path in sorted(masks.values(), key=lambda path: path.name.lower())
    )
    assert event[2:] == (str(m2m), str(scratch), 0.5, False)
    assert len(result.surfaces) == 6
    assert len(result.relabel_surfaces) == 1
    assert result.vertex_density == 0.5
    assert all(path.is_file() for path in (*result.surfaces, *result.relabel_surfaces))


def test_volume_meshing_follows_official_cleanup_order_and_validates(
    tmp_path: Path,
) -> None:
    events: list[tuple] = []
    api = _recording_api(tmp_path, events)
    m2m = tmp_path / "m2m"
    m2m.mkdir()
    surfaces = []
    for name in ("wm", "gm", "csf", "bone", "skin", "eyes", "ventricles"):
        path = m2m / f"{name}.stl"
        path.write_bytes(name.encode())
        surfaces.append(path)
    relabel = [m2m / "air_outer_relabel_1.stl"]
    relabel[0].write_bytes(b"air")
    output = tmp_path / "subject.msh"

    validation = build_volume_mesh(
        api,
        "sub-test",
        surfaces,
        relabel,
        m2m,
        output,
        keep_air=False,
    )

    names = [event[0] for event in events]
    expected = [
        "make_volume_mesh",
        "read_msh_raw",
        "relabel_compartments",
        "volume_mesh_relabel",
        "fix_surface_labels",
        "fix_thin_tetrahedra",
        "fix_th_node_ordering",
        "fix_tr_node_ordering",
        "write_msh",
        "read_msh_final",
        "elements_volumes_and_areas",
    ]
    positions = [names.index(name) for name in expected]
    assert positions == sorted(positions)
    make_event = next(event for event in events if event[0] == "make_volume_mesh")
    assert make_event == (
        "make_volume_mesh",
        "sub-test",
        tuple(str(path) for path in sorted(surfaces, key=lambda path: path.name.lower())),
        str(m2m),
        False,
    )
    read_events = [event for event in events if event[0].startswith("read_msh")]
    assert all(event[2] is True for event in read_events)
    assert validation.tissue_tags == (1, 2, 3, 4, 5, 6)
    assert validation.tetrahedra == 6
    assert validation.minimum_tetrahedron_volume == 0.5
    assert output.is_file()
    assert not (m2m / "sub-test.msh").exists()


def test_volume_cleanup_skips_optional_relabels_when_not_applicable(tmp_path: Path) -> None:
    events: list[tuple] = []
    api = _recording_api(tmp_path, events)
    m2m = tmp_path / "m2m"
    m2m.mkdir()
    surfaces = []
    for name in ("wm", "gm", "csf", "bone", "skin", "eyes"):
        path = m2m / f"{name}.stl"
        path.write_bytes(name.encode())
        surfaces.append(path)

    build_volume_mesh(api, "sub-test", surfaces, [], m2m, tmp_path / "final.msh")

    names = [event[0] for event in events]
    assert "relabel_compartments" not in names
    assert "volume_mesh_relabel" not in names


@pytest.mark.parametrize(
    ("tags", "volumes", "message"),
    [
        ((1, 2, 3, 4, 5), (1, 1, 1, 1, 1), "tissue tags"),
        ((1, 2, 3, 4, 5, 6), (1, 1, 0, 1, 1, 1), "positive"),
        ((1, 2, 3, 4, 5, 6), (1, 1, np.nan, 1, 1, 1), "finite"),
    ],
)
def test_tag_and_positive_volume_validation_fails_closed(
    tmp_path: Path,
    tags: tuple[int, ...],
    volumes: tuple[float, ...],
    message: str,
) -> None:
    events: list[tuple] = []
    api = _recording_api(tmp_path, events, tags=tags, volumes=volumes)
    mesh = tmp_path / "invalid.msh"
    mesh.write_bytes(b"mesh")

    with pytest.raises(MeshValidationError, match=message):
        validate_mesh(api, mesh)


def test_raw_gmsh_mesh_is_kept_when_final_validation_fails(tmp_path: Path) -> None:
    events: list[tuple] = []
    api = _recording_api(tmp_path, events, tags=(1, 2, 3, 4, 5))
    m2m = tmp_path / "m2m"
    m2m.mkdir()
    surfaces = []
    for name in ("wm", "gm", "csf", "bone", "skin", "eyes"):
        path = m2m / f"{name}.stl"
        path.write_bytes(name.encode())
        surfaces.append(path)

    with pytest.raises(MeshValidationError, match="tissue tags"):
        build_volume_mesh(api, "sub-test", surfaces, [], m2m, tmp_path / "final.msh")

    assert (m2m / "sub-test.msh").is_file()


def _recording_api(
    tmp_path: Path,
    events: list[tuple],
    *,
    tags: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    volumes: tuple[float, ...] = (0.5, 1, 1, 1, 1, 1),
) -> V3MeshingAPI:
    class RecordingMesh:
        def __init__(self) -> None:
            self.fn = ""
            self.elm = SimpleNamespace(
                elm_type=np.asarray([4] * len(tags), dtype=int),
                tag1=np.asarray(tags, dtype=int),
            )

        def fix_surface_labels(self) -> None:
            events.append(("fix_surface_labels",))

        def fix_thin_tetrahedra(self) -> None:
            events.append(("fix_thin_tetrahedra",))

        def fix_th_node_ordering(self) -> None:
            events.append(("fix_th_node_ordering",))

        def fix_tr_node_ordering(self) -> None:
            events.append(("fix_tr_node_ordering",))

        def elements_volumes_and_areas(self):
            events.append(("elements_volumes_and_areas",))
            return SimpleNamespace(value=np.asarray(volumes, dtype=float))

    raw_mesh = tmp_path / "m2m" / "sub-test.msh"

    def recording_surfaces(
        input_files,
        out_dir,
        temp_dir,
        vertex_density=0.5,
        catsurf=False,
    ):
        events.append(
            (
                "make_surface_meshes",
                tuple(input_files),
                out_dir,
                temp_dir,
                vertex_density,
                catsurf,
            )
        )
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        surfaces = []
        for name in ("wm", "gm", "csf", "bone", "skin", "eyes"):
            path = out / f"{name}.stl"
            path.write_bytes(name.encode())
            surfaces.append(str(path))
        relabel = out / "air_outer_relabel_1.stl"
        relabel.write_bytes(b"air")
        return surfaces, [str(relabel)]

    def recording_volume(subject_id, input_files, out_dir, keep_air):
        events.append(
            (
                "make_volume_mesh",
                subject_id,
                tuple(input_files),
                out_dir,
                keep_air,
            )
        )
        raw_mesh.parent.mkdir(parents=True, exist_ok=True)
        raw_mesh.write_bytes(b"raw")
        return str(raw_mesh)

    def recording_read(filename, m=None, buffered=False, skip_data=False):
        del m, skip_data
        kind = "raw" if Path(filename) == raw_mesh else "final"
        events.append((f"read_msh_{kind}", str(filename), buffered))
        return RecordingMesh()

    def recording_write(mesh, filename=None, mode="binary"):
        del mode
        events.append(("write_msh", mesh.fn, str(filename)))
        Path(filename).write_bytes(b"final")

    def recording_relabel(mesh, old, new):
        events.append(("relabel_compartments", tuple(old), tuple(new)))
        return mesh

    def recording_volume_relabel(mesh, relabel_surfaces, region_number):
        events.append(
            ("volume_mesh_relabel", tuple(relabel_surfaces), region_number)
        )
        return mesh

    return V3MeshingAPI(
        simnibs=SimpleNamespace(__version__="3.2.6", __file__="/fake/simnibs.py"),
        headreco=SimpleNamespace(
            make_surface_meshes=recording_surfaces,
            make_volume_mesh=recording_volume,
            __file__="/fake/headreco.py",
        ),
        hmutils=SimpleNamespace(
            relabel_compartments=recording_relabel,
            volume_mesh_relabel=recording_volume_relabel,
            path2bin=lambda name: f"/fake/{name}",
            __file__="/fake/hmutils.py",
        ),
        mesh_io=SimpleNamespace(
            read_msh=recording_read,
            write_msh=recording_write,
            Msh=RecordingMesh,
            __file__="/fake/mesh_io.py",
        ),
    )
