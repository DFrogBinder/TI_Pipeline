import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ROOT_DIR = Path("/home/uos/Boyan/")
DEFAULT_DATA_DIR = DEFAULT_ROOT_DIR / "CamCan_Data"
DEFAULT_OUTPUT_DIR_NAME = "FastSurfer_out"
DEFAULT_THREADS_PER_JOB = 9
DEFAULT_MAX_PARALLEL_JOBS = 3
DEFAULT_LICENSE_PATH = DEFAULT_ROOT_DIR / "freesurfer_licence.txt"

SCRIPT_DIR = Path(__file__).resolve().parent


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, received: {value}") from exc
    if parsed < 1:
        raise SystemExit(f"{name} must be a positive integer, received: {value}")
    return parsed


def parse_subjects(raw: str) -> list[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def load_subjects_file(path: Path) -> list[str]:
    subjects: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            subjects.extend(parse_subjects(stripped))
    return subjects


def unique_sorted(subjects: list[str]) -> list[str]:
    return sorted(dict.fromkeys(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch FastSurfer/FreeSurfer atlas creation across CamCAN subjects."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("ATLAS_DATA_DIR", DEFAULT_DATA_DIR)),
        help="Dataset root containing <subject>/anat/<subject>_T1w.nii.gz.",
    )
    parser.add_argument(
        "--license-path",
        type=Path,
        default=Path(os.environ.get("ATLAS_LICENSE_PATH", DEFAULT_LICENSE_PATH)),
        help="FreeSurfer license file.",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=env_int("ATLAS_THREADS_PER_JOB", DEFAULT_THREADS_PER_JOB),
        help="CPU threads passed to FastSurfer/FreeSurfer for each subject.",
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=env_int("ATLAS_MAX_PARALLEL_JOBS", DEFAULT_MAX_PARALLEL_JOBS),
        help="Number of subjects processed concurrently.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=os.environ.get("ATLAS_OUTPUT_DIR_NAME", DEFAULT_OUTPUT_DIR_NAME),
        help="Directory under data-dir where FastSurfer outputs are written.",
    )
    parser.add_argument(
        "--subjects",
        default=os.environ.get("ATLAS_SUBJECTS", ""),
        help="Optional comma/space-separated subject IDs to run.",
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=Path(os.environ["ATLAS_SUBJECTS_FILE"])
        if os.environ.get("ATLAS_SUBJECTS_FILE")
        else None,
        help="Optional text file of subject IDs, one or more per line.",
    )
    return parser.parse_args()


def input_available(data_dir: Path, subject: str) -> bool:
    """Return True if the subject has the expected anatomical input."""
    t1_file = data_dir / subject / "anat" / f"{subject}_T1w.nii.gz"
    return t1_file.is_file()


def already_processed(output_dir: Path, subject: str) -> bool:
    """Return True if the subject already has the final NIfTI output."""
    nifti = output_dir / subject / "mri" / "aparc.DKTatlas+aseg.deep.nii.gz"
    return nifti.is_file()


def discover_subjects(data_dir: Path, output_dir_name: str) -> list[str]:
    return sorted(
        entry.name
        for entry in data_dir.iterdir()
        if entry.is_dir() and entry.name != output_dir_name
    )


def configured_subjects(args: argparse.Namespace, data_dir: Path) -> list[str]:
    subjects = parse_subjects(args.subjects)
    if args.subjects_file:
        subjects.extend(load_subjects_file(args.subjects_file.expanduser()))

    if subjects:
        return unique_sorted(subjects)

    return discover_subjects(data_dir, args.output_dir_name)


def process_subject(
    data_dir: Path,
    threads_per_job: int,
    license_path: Path,
    subject: str,
) -> None:
    """Run make_atlas.sh for a single subject."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_job)

    cmd = [
        str(SCRIPT_DIR / "make_atlas.sh"),
        str(data_dir),
        str(threads_per_job),
        str(license_path),
        subject,
    ]
    print(f"[run] Processing {subject} ...")
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    license_path = args.license_path.expanduser().resolve()
    output_dir = data_dir / args.output_dir_name

    if not data_dir.is_dir():
        raise SystemExit(f"DATA_DIR not found: {data_dir}")
    if not license_path.is_file():
        raise SystemExit(f"LICENSE_PATH not found: {license_path}")
    if args.threads_per_job < 1:
        raise SystemExit("--threads-per-job must be positive")
    if args.max_parallel_jobs < 1:
        raise SystemExit("--max-parallel-jobs must be positive")

    subject_list = configured_subjects(args, data_dir)
    futures = {}

    print(f"[config] DATA_DIR: {data_dir}")
    print(f"[config] OUTPUT_DIR: {output_dir}")
    print(f"[config] LICENSE_PATH: {license_path}")
    print(f"[config] THREADS_PER_JOB: {args.threads_per_job}")
    print(f"[config] MAX_PARALLEL_JOBS: {args.max_parallel_jobs}")
    print(f"[config] SUBJECTS_DISCOVERED: {len(subject_list)}")

    with ThreadPoolExecutor(max_workers=args.max_parallel_jobs) as executor:
        for subject in subject_list:
            if subject == args.output_dir_name:
                print(f"[skip] '{subject}' is the output directory.")
                continue
            if not input_available(data_dir, subject):
                print(f"[skip] Missing required T1w input for {subject}.")
                continue
            if already_processed(output_dir, subject):
                print(f"[skip] {subject} already processed.")
                continue

            futures[
                executor.submit(
                    process_subject,
                    data_dir,
                    args.threads_per_job,
                    license_path,
                    subject,
                )
            ] = subject

        if not futures:
            print("[done] No subjects needed processing.")
            return 0

        for future in as_completed(futures):
            subject = futures[future]
            try:
                future.result()
                print(f"[done] {subject}")
            except subprocess.CalledProcessError as exc:
                print(f"[error] {subject} failed with exit code {exc.returncode}")
                raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
