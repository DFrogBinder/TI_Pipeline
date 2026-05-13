#!/usr/bin/env python3
"""
Submit a one-subject FreeSurfer MNI152 Destrieux atlas job.

This helper keeps the submission command short while the Slurm script does the
actual recon-all and mri_convert work.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SLURM = DEFAULT_REPO_DIR / "HPC_scripts" / "run_mni152_atlas.slurm"
DEFAULT_SUBJECTS_DIR = Path("/mnt/parscratch/users/cop23bi/ti_dataset/derivatives/freesurfer")
DEFAULT_DEST = Path("/mnt/parscratch/users/cop23bi/atlases_new")
DEFAULT_LICENSE = Path("/users/cop23bi/freesurfer_licence.txt")

SOURCE_ATLAS_CHOICES = {
    "destrieux": "aparc.a2009s+aseg.mgz",
    "dkt": "aparc.DKTatlas+aseg.mgz",
    "desikan": "aparc+aseg.mgz",
    "aseg": "aseg.mgz",
}


def existing_file(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def path_arg(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_sbatch_command(args: argparse.Namespace) -> list[str]:
    source_atlas = SOURCE_ATLAS_CHOICES[args.atlas]
    export_values = {
        "MNI_T1": str(args.mni_t1),
        "MNI_SUBJECT_ID": args.subject_id,
        "SUBJECTS_DIR": str(args.subjects_dir),
        "MNI_ATLAS_DEST": str(args.dest),
        "FS_LICENSE": str(args.fs_license),
        "FREESURFER_MODULE": args.freesurfer_module,
        "SOURCE_ATLAS": source_atlas,
        "OUTPUT_FILENAME": args.output_filename or f"{args.subject_id}.nii.gz",
    }

    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    command = ["sbatch", f"--export={export_arg}"]

    if args.partition:
        command.append(f"--partition={args.partition}")
    if args.cpus:
        command.append(f"--cpus-per-task={args.cpus}")
    if args.mem:
        command.append(f"--mem={args.mem}")
    if args.time:
        command.append(f"--time={args.time}")

    command.append(str(args.slurm_script))
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the one-subject MNI152 FreeSurfer Destrieux atlas job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mni-t1",
        required=True,
        type=existing_file,
        help="MNI152 T1 image to pass to recon-all.",
    )
    parser.add_argument(
        "--subjects-dir",
        type=path_arg,
        default=DEFAULT_SUBJECTS_DIR,
        help="FreeSurfer SUBJECTS_DIR where sub-mni152 will be created.",
    )
    parser.add_argument(
        "--dest",
        type=path_arg,
        default=DEFAULT_DEST,
        help="Flat destination atlas directory.",
    )
    parser.add_argument(
        "--fs-license",
        type=existing_file,
        default=DEFAULT_LICENSE,
        help="FreeSurfer license file.",
    )
    parser.add_argument(
        "--subject-id",
        default="sub-mni152",
        help="FreeSurfer subject ID to create or resume.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Output filename inside --dest. Defaults to <subject-id>.nii.gz.",
    )
    parser.add_argument(
        "--atlas",
        choices=sorted(SOURCE_ATLAS_CHOICES),
        default="destrieux",
        help="Which FreeSurfer atlas MGZ to convert after recon-all.",
    )
    parser.add_argument(
        "--slurm-script",
        type=existing_file,
        default=DEFAULT_SLURM,
        help="Parameterized Slurm script to submit.",
    )
    parser.add_argument(
        "--freesurfer-module",
        default="FreeSurfer/7.4.1-centos7_x86_64",
        help="HPC module loaded by the Slurm script.",
    )
    parser.add_argument("--partition", default=None, help="Override Slurm partition.")
    parser.add_argument("--cpus", type=int, default=None, help="Override Slurm cpus-per-task.")
    parser.add_argument("--mem", default=None, help="Override Slurm memory, e.g. 32G.")
    parser.add_argument("--time", default=None, help="Override Slurm walltime, e.g. 24:00:00.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually run sbatch. Without this, print the command only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_sbatch_command(args)

    print(shell_join(command))
    if not args.submit:
        print("\nDry run only. Re-run with --submit to submit this job.")
        return 0

    env = os.environ.copy()
    result = subprocess.run(command, check=False, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
