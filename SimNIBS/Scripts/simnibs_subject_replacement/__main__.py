"""Command-line interface for the replacement-subject orchestration package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replacement import (
    load_prepared_manifest,
    preflight,
    prepare,
    require_left_complete,
    require_v3_promotion,
)
from .settings import (
    DEFAULT_ATLAS_ROOT,
    DEFAULT_CORRECTED_V4_SCAFFOLD,
    DEFAULT_FINAL132_SUBJECTS,
    DEFAULT_TARGETS_CSV,
    DEFAULT_WORK_ROOT,
    ReplacementConfig,
    replacement_paths,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subject", required=True)
    parser.add_argument("--replaces", required=True)
    parser.add_argument("--legacy-v3-root", type=Path, required=True)
    parser.add_argument(
        "--final132-subjects", type=Path, default=DEFAULT_FINAL132_SUBJECTS
    )
    parser.add_argument(
        "--corrected-v4-scaffold",
        type=Path,
        default=DEFAULT_CORRECTED_V4_SCAFFOLD,
    )
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)


def _config(args: argparse.Namespace) -> ReplacementConfig:
    return ReplacementConfig(
        subject=args.subject,
        replaces=args.replaces,
        legacy_v3_root=args.legacy_v3_root,
        final132_subjects_file=args.final132_subjects,
        corrected_v4_scaffold_root=args.corrected_v4_scaffold,
        atlas_root=args.atlas_root,
        targets_csv=args.targets_csv,
        work_root=args.work_root,
    )


def _manifest(config: ReplacementConfig) -> dict[str, object]:
    manifest = load_prepared_manifest(replacement_paths(config).manifest)
    if manifest.get("subject") != config.subject or manifest.get("replaces") != config.replaces:
        raise RuntimeError("Prepared manifest is bound to different runtime identity")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare", "bindings", "require-promotion", "status"):
        _add_common(commands.add_parser(name))
    left = commands.add_parser("require-left")
    _add_common(left)
    left.add_argument("--version", choices=("v3", "v4"), required=True)
    v4_init = commands.add_parser("v4-init")
    _add_common(v4_init)
    v4_init.add_argument(
        "--target", choices=("left-hippocampus", "right-m1"), required=True
    )
    v4_run = commands.add_parser("v4-run")
    _add_common(v4_run)
    v4_run.add_argument(
        "--target", choices=("left-hippocampus", "right-m1"), required=True
    )
    v4_run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    if args.command == "preflight":
        payload: object = preflight(config)
    elif args.command == "prepare":
        payload = prepare(config)
    else:
        manifest = _manifest(config)
        if args.command == "bindings":
            paths = manifest["paths"]
            assert isinstance(paths, dict)
            inputs = manifest["inputs"]
            assert isinstance(inputs, dict)
            atlas = inputs["atlas"]
            targets_csv = inputs["targets_csv"]
            assert isinstance(atlas, dict) and isinstance(targets_csv, dict)
            values = (
                paths["manifest"],
                paths["v4_corrected_root"],
                paths["v3_headmodel_root"],
                paths["v3_left_root"],
                paths["v3_right_root"],
                paths["v4_left_root"],
                paths["v4_right_root"],
                config.legacy_v3_root.resolve(),
                Path(str(atlas["path"])).parent,
                targets_csv["path"],
            )
            print("\n".join(str(value) for value in values))
            return 0
        if args.command == "require-promotion":
            payload = require_v3_promotion(manifest)
        elif args.command == "require-left":
            payload = require_left_complete(manifest, args.version)
        elif args.command == "v4-init":
            from .v4_workflow import initialize_v4_target

            return initialize_v4_target(manifest, args.target)
        elif args.command == "v4-run":
            from .v4_workflow import run_v4_target

            return run_v4_target(manifest, args.target, args.dry_run)
        else:
            from .v4_workflow import replacement_status

            payload = replacement_status(manifest)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
