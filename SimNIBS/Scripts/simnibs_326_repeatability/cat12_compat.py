#!/usr/bin/env python3
"""Bounded MATLAB R2023b compatibility for SimNIBS 3.2.6 CAT12.

The CAT12 revision bundled with SimNIBS 3.2.6 writes a MAT report and then
raises when its legacy XML serializer fails under newer MATLAB releases.  The
MAT report is intact and headreco does not consume the XML report.  Build a
temporary overlay from hash-verified upstream files that preserves the MAT
write and downgrades only the two XML-write failures to warnings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PATCH_ID = "simnibs326-cat12-r2023b-xml-warning-v1"
EXPECTED_SOURCE_SHA256 = {
    "segment_CAT.m": "234f908b99031453807e92aeec14445e4bf3e3188c76f3fbb0cbe2222d37c970",
    "cat_io_xml.m": "7f64ad2ad71123fbfc67c0f5abd29519d5a9b24428c881ae1da5099aee8cd45e",
}
XML_WRITE_ERROR = (
    "        error('MATLAB:cat_io_xml:writeErr','Can''t write XML-file "
    "''%s''!\\n',file);"
)
XML_WRITE_WARNING = (
    "        warning('MATLAB:cat_io_xml:writeErr','Can''t write XML-file "
    "''%s''!\\n',file);"
)
SEGMENT_INSERTION_POINT = "end\ncat_get_defaults('output.CSF.native', true);"
SEGMENT_COMPAT_BLOCK = """end

% SimNIBS 3.2.6 compatibility overlay for MATLAB R2023b.
compat_dir = getenv('SIMNIBS326_CAT12_COMPAT_DIR');
if isempty(compat_dir)
    error('SIMNIBS326:CAT12Compat', ...
        'SIMNIBS326_CAT12_COMPAT_DIR is not set.');
end
addpath(compat_dir,'-begin');
rehash;
expected_cat_io_xml = fullfile(compat_dir,'cat_io_xml.m');
observed_cat_io_xml = which('cat_io_xml');
if ~strcmp(observed_cat_io_xml,expected_cat_io_xml)
    error('SIMNIBS326:CAT12Compat', ...
        'CAT12 compatibility overlay is not first on the MATLAB path.');
end
fprintf('SIMNIBS326_CAT12_COMPAT_ACTIVE %s\\n',observed_cat_io_xml);

cat_get_defaults('output.CSF.native', true);"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _installed_sources() -> dict[str, Path]:
    import simnibs  # type: ignore

    package_root = Path(simnibs.__file__).resolve().parent
    return {
        "segment_CAT.m": package_root / "msh" / "segment_CAT.m",
        "cat_io_xml.m": (
            package_root
            / "resources"
            / "spm12"
            / "toolbox"
            / "cat12"
            / "cat_io_xml.m"
        ),
    }


def _patch_source_text(
    *, segment_text: str, xml_text: str
) -> tuple[str, str, dict[str, int]]:
    xml_replacements = xml_text.count(XML_WRITE_ERROR)
    if xml_replacements != 2:
        raise RuntimeError(
            "Expected exactly two fatal CAT12 XML-write branches, observed "
            f"{xml_replacements}"
        )
    patched_xml = xml_text.replace(XML_WRITE_ERROR, XML_WRITE_WARNING)

    segment_replacements = segment_text.count(SEGMENT_INSERTION_POINT)
    if segment_replacements != 1:
        raise RuntimeError(
            "Expected exactly one segment_CAT compatibility insertion point, "
            f"observed {segment_replacements}"
        )
    patched_segment = segment_text.replace(
        SEGMENT_INSERTION_POINT,
        SEGMENT_COMPAT_BLOCK,
        1,
    )
    return patched_segment, patched_xml, {
        "segment_path_insertions": segment_replacements,
        "xml_error_to_warning_replacements": xml_replacements,
    }


def build_overlay(destination: Path) -> dict[str, object]:
    sources = _installed_sources()
    observed_hashes: dict[str, str] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing SimNIBS 3.2.6 source file: {source}")
        observed_hashes[name] = _sha256(source)
        expected = EXPECTED_SOURCE_SHA256[name]
        if observed_hashes[name] != expected:
            raise RuntimeError(
                f"Refusing to patch unexpected {name}: "
                f"observed={observed_hashes[name]}; expected={expected}; "
                f"path={source}"
            )

    patched_segment, patched_xml, replacements = _patch_source_text(
        segment_text=sources["segment_CAT.m"].read_text(encoding="utf-8"),
        xml_text=sources["cat_io_xml.m"].read_text(encoding="utf-8"),
    )
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {
        "segment_CAT.m": destination / "segment_CAT.m",
        "cat_io_xml.m": destination / "cat_io_xml.m",
    }
    outputs["segment_CAT.m"].write_text(patched_segment, encoding="utf-8")
    outputs["cat_io_xml.m"].write_text(patched_xml, encoding="utf-8")
    output_hashes = {name: _sha256(path) for name, path in outputs.items()}
    return {
        "patch_id": PATCH_ID,
        "status": "prepared",
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_sha256": observed_hashes,
        "patched_paths": {name: str(path) for name, path in outputs.items()},
        "patched_sha256": output_hashes,
        "replacements": replacements,
    }


@contextmanager
def temporary_overlay() -> Iterator[tuple[Path, dict[str, object]]]:
    parent = os.environ.get("TMPDIR") or os.environ.get("SLURM_TMPDIR")
    temporary = Path(
        tempfile.mkdtemp(
            prefix="simnibs326_cat12_",
            dir=parent,
        )
    )
    overlay = temporary / "overlay"
    try:
        yield overlay, build_overlay(overlay)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _inject_matlab_overlay(command: str, overlay: Path) -> str:
    if "segment_CAT(" not in command:
        return command
    needle = ";try,"
    if command.count(needle) != 1:
        raise RuntimeError(
            "Refusing to modify an unexpected headreco MATLAB command: "
            f"found {command.count(needle)} injection points"
        )
    injected = f";addpath('{_matlab_quote(overlay)}','-begin');try,"
    return command.replace(needle, injected, 1)


def run_headreco(arguments: list[str]) -> int:
    from simnibs.msh import headreco as upstream_headreco  # type: ignore

    original_spawn = upstream_headreco.hmu.spawn_process
    injection_count = 0
    with temporary_overlay() as (overlay, metadata):
        print(
            json.dumps(
                {"event": "cat12_compat_overlay_ready", **metadata},
                sort_keys=True,
            ),
            flush=True,
        )
        previous_overlay = os.environ.get("SIMNIBS326_CAT12_COMPAT_DIR")
        os.environ["SIMNIBS326_CAT12_COMPAT_DIR"] = str(overlay)

        def compat_spawn(command, *args, **kwargs):
            nonlocal injection_count
            if isinstance(command, str) and "segment_CAT(" in command:
                command = _inject_matlab_overlay(command, overlay)
                injection_count += 1
                print(
                    json.dumps(
                        {
                            "event": "cat12_compat_matlab_path_injected",
                            "patch_id": PATCH_ID,
                        }
                    ),
                    flush=True,
                )
            return original_spawn(command, *args, **kwargs)

        upstream_headreco.hmu.spawn_process = compat_spawn
        try:
            result = upstream_headreco.headmodel(["headreco", *arguments])
        finally:
            upstream_headreco.hmu.spawn_process = original_spawn
            if previous_overlay is None:
                os.environ.pop("SIMNIBS326_CAT12_COMPAT_DIR", None)
            else:
                os.environ["SIMNIBS326_CAT12_COMPAT_DIR"] = previous_overlay

        if arguments and arguments[0] in ("all", "preparevols"):
            if injection_count != 1:
                raise RuntimeError(
                    "CAT12 compatibility path was not injected exactly once: "
                    f"observed={injection_count}"
                )
        return int(result or 0)


def probe_matlab_compatibility(matlab: str) -> dict[str, object]:
    marker = "SIMNIBS326_CAT12_COMPAT_PROBE_OK"
    with temporary_overlay() as (overlay, metadata):
        probe_parent = overlay.parent / "probe"
        probe_parent.mkdir()
        probe_xml = probe_parent / "cat12_compat_probe.xml"
        probe_mat = probe_parent / "cat12_compat_probe.mat"
        script = ";".join(
            (
                f"addpath('{_matlab_quote(overlay)}','-begin')",
                (
                    "assert(strcmp(which('cat_io_xml'),"
                    f"fullfile('{_matlab_quote(overlay)}','cat_io_xml.m')))"
                ),
                "S=struct('probe',1)",
                f"cat_io_xml('{_matlab_quote(probe_xml)}',S,'write')",
                f"assert(exist('{_matlab_quote(probe_mat)}','file')==2)",
                f"disp('{marker}')",
            )
        )
        try:
            result = subprocess.run(
                [matlab, "-batch", script],
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
                env={
                    **os.environ,
                    "SIMNIBS326_CAT12_COMPAT_DIR": str(overlay),
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "CAT12 compatibility MATLAB probe exceeded 300 seconds"
            ) from exc
        output = result.stdout + result.stderr
        if result.returncode != 0 or marker not in output or not probe_mat.is_file():
            raise RuntimeError(
                "CAT12 compatibility MATLAB probe failed: "
                f"returncode={result.returncode}; output_tail={output[-4000:]}"
            )
        return {
            **metadata,
            "status": "ready",
            "matlab_command": [matlab, "-batch", "<cat12-compat-probe>"],
            "matlab_returncode": result.returncode,
            "mat_report_created": True,
            "xml_report_created": probe_xml.is_file(),
            "output_tail": output[-2000:],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    headreco = subparsers.add_parser("headreco")
    headreco.add_argument("arguments", nargs=argparse.REMAINDER)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--matlab", default="matlab")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "headreco":
        return run_headreco(args.arguments)
    payload = probe_matlab_compatibility(args.matlab)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
