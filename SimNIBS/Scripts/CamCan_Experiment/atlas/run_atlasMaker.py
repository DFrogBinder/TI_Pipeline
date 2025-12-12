import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT_DIR = Path("/home/uos/Boyan/")
DATA_DIR = ROOT_DIR / "CamCan_Data"
OUTPUT_DIR_NAME = "FastSurfer_out"
OUTPUT_DIR = DATA_DIR / OUTPUT_DIR_NAME
THREADS_PER_JOB = 9
MAX_PARALLEL_JOBS = 3
LICENSE_PATH = ROOT_DIR / "freesurfer_licence.txt"

SCRIPT_DIR = Path(__file__).resolve().parent


def input_available(subject: str) -> bool:
    """Return True if the subject has the expected anatomical input."""
    t1_file = DATA_DIR / subject / "anat" / f"{subject}_T1w.nii.gz"
    return t1_file.is_file()


def already_processed(subject: str) -> bool:
    """Return True if the subject already has the final NIfTI output."""
    nifti = OUTPUT_DIR / subject / "mri" / "aparc.DKTatlas+aseg.deep.nii.gz"
    return nifti.is_file()


def process_subject(subject: str) -> None:
    """Run make_atlas.sh for a single subject."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS_PER_JOB)

    cmd = [
        str(SCRIPT_DIR / "make_atlas.sh"),
        str(DATA_DIR),
        subject,
        str(THREADS_PER_JOB),
        str(LICENSE_PATH),
    ]
    print(f"[run] Processing {subject} ...")
    subprocess.run(cmd, env=env, check=True)


subject_list = sorted(entry.name for entry in DATA_DIR.iterdir() if entry.is_dir())
futures = {}

with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
    for subject in subject_list:
        if subject == OUTPUT_DIR_NAME:
            print(f"[skip] '{subject}' is the output directory.")
            continue
        if not input_available(subject):
            print(f"[skip] Missing required T1w input for {subject}.")
            continue
        if already_processed(subject):
            print(f"[skip] {subject} already processed.")
            continue

        futures[executor.submit(process_subject, subject)] = subject

    for future in as_completed(futures):
        subject = futures[future]
        try:
            future.result()
            print(f"[done] {subject}")
        except subprocess.CalledProcessError as exc:
            print(f"[error] {subject} failed with exit code {exc.returncode}")
            raise
