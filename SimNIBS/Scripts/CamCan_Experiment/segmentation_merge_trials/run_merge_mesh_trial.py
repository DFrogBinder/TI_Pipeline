#!/usr/bin/env python3
"""Run CHARM meshing trials for two custom/CHARM segmentation merge strategies.

This runner deliberately stops after CHARM meshing. It does not run SimNIBS
electric-field simulations.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


TIMEOUT_EXIT_CODE = 124
INPUT_EXIT_CODE = 126
INCOMPLETE_EXIT_CODE = 125

SKIN_LABEL = 5
BACKGROUND_LABEL = 0

STRATEGIES = {
    "candidate_a": {
        "directory": "candidate_A_full_charm_fallback",
        "description": "Full CHARM tissue map fallback plus manual positive non-skin overlay.",
    },
    "candidate_b": {
        "directory": "candidate_B_solid_charm_head_skin_base",
        "description": "Solid CHARM head envelope labelled as skin plus manual positive non-skin overlay.",
    },
}


class CommandTimeout(RuntimeError):
    def __init__(self, label: str, timeout_sec: float | None):
        super().__init__(f"{label} timed out after {timeout_sec:.0f} seconds")
        self.label = label
        self.timeout_sec = timeout_sec


class MissingInput(RuntimeError):
    pass


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def kill_process_group(process: subprocess.Popen[str], *, label: str, sig: int, name: str) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
        log_event("cmd_signal", label=label, signal=name, pid=process.pid)
    except ProcessLookupError:
        pass


def run_cmd(cmd: list[str], *, cwd: Path, label: str, deadline: float | None) -> None:
    timeout_sec = remaining_timeout(deadline)
    if timeout_sec is not None and timeout_sec <= 0:
        raise CommandTimeout(label, timeout_sec)

    log_event("run_cmd", label=label, cmd=cmd, cwd=str(cwd), timeout_sec=timeout_sec)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        kill_process_group(process, label=label, sig=signal.SIGTERM, name="SIGTERM")
        try:
            extra_stdout, extra_stderr = process.communicate(timeout=30)
            stdout += extra_stdout or ""
            stderr += extra_stderr or ""
        except subprocess.TimeoutExpired:
            kill_process_group(process, label=label, sig=signal.SIGKILL, name="SIGKILL")
            extra_stdout, extra_stderr = process.communicate()
            stdout += extra_stdout or ""
            stderr += extra_stderr or ""

        log_event(
            "cmd_timeout",
            label=label,
            returncode=process.returncode,
            stdout_tail=stdout[-4000:] if stdout else "",
            stderr_tail=stderr[-4000:] if stderr else "",
        )
        raise CommandTimeout(label, timeout_sec) from exc

    log_event(
        "cmd_result",
        label=label,
        returncode=process.returncode,
        stdout_tail=stdout[-4000:] if stdout else "",
        stderr_tail=stderr[-4000:] if stderr else "",
    )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)


def candidate_nifti_paths(anat_dir: Path, stem: str) -> tuple[Path, Path]:
    return anat_dir / f"{stem}.nii", anat_dir / f"{stem}.nii.gz"


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def source_anat_dir(source_root: Path, subject: str) -> Path:
    direct = source_root
    nested = source_root / subject / "anat"
    if nested.is_dir():
        return nested
    if direct.name == "anat" and direct.is_dir():
        return direct
    raise MissingInput(f"Could not find anat directory for {subject} under {source_root}")


def resolve_source_inputs(source_root: Path, subject: str) -> dict[str, Path]:
    anat = source_anat_dir(source_root, subject)
    t1 = first_existing(candidate_nifti_paths(anat, f"{subject}_T1w"))
    t2 = first_existing(candidate_nifti_paths(anat, f"{subject}_T2w"))
    manual = first_existing(candidate_nifti_paths(anat, f"{subject}_T1w_ras_1mm_T1andT2_masks"))

    missing = []
    if t1 is None:
        missing.append(f"{subject}_T1w.nii[.gz]")
    if t2 is None:
        missing.append(f"{subject}_T2w.nii[.gz]")
    if manual is None:
        missing.append(f"{subject}_T1w_ras_1mm_T1andT2_masks.nii[.gz]")
    if missing:
        raise MissingInput(f"Missing required inputs in {anat}: {', '.join(missing)}")

    return {"anat": anat, "t1": t1, "t2": t2, "manual": manual}


def copy_input_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def subject_suffix(subject: str) -> str:
    return subject.split("-")[-1].upper()


def m2m_candidates(anat_dir: Path, subject: str) -> list[Path]:
    suffix = subject_suffix(subject)
    return [
        anat_dir / f"m2m_{subject}",
        anat_dir / f"m2m_sub-{suffix}",
    ]


def find_m2m_dir(anat_dir: Path, subject: str) -> Path:
    for candidate in m2m_candidates(anat_dir, subject):
        if candidate.is_dir():
            return candidate
    matches = sorted(anat_dir.glob("m2m_*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No m2m directory found under {anat_dir}")


def find_charm_label(anat_dir: Path, subject: str) -> Path:
    for m2m_dir in m2m_candidates(anat_dir, subject):
        label = m2m_dir / "label_prep" / "tissue_labeling_upsampled.nii.gz"
        if label.is_file():
            return label
    matches = sorted(anat_dir.glob("m2m_*/label_prep/tissue_labeling_upsampled.nii.gz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"CHARM tissue_labeling_upsampled.nii.gz not found under {anat_dir}")


def find_mesh(anat_dir: Path, subject: str) -> Path | None:
    suffix = subject_suffix(subject)
    candidates = [
        anat_dir / f"m2m_{subject}" / f"{subject}.msh",
        anat_dir / f"m2m_sub-{suffix}" / f"{subject}.msh",
        anat_dir / f"m2m_sub-{suffix}" / f"sub-{suffix}.msh",
    ]
    found = first_existing(candidates)
    if found is not None:
        return found
    matches = sorted(anat_dir.glob("m2m_*/*.msh"))
    return matches[0] if matches else None


def base_marker(base_root: Path) -> Path:
    return base_root / "BASE_COMPLETE.json"


def strategy_marker(strategy_root: Path) -> Path:
    return strategy_root / "COMPLETE.json"


def prepare_base_charm(
    *,
    source_root: Path,
    out_root: Path,
    subject: str,
    force: bool,
    deadline: float | None,
) -> Path:
    subject_root = out_root / subject
    base_root = subject_root / "charm_base"
    base_anat = base_root / "anat"
    marker = base_marker(base_root)

    if force and base_root.exists():
        shutil.rmtree(base_root)

    if marker.is_file():
        try:
            payload = read_json(marker)
            label_path = Path(payload["charm_label_path"])
            if label_path.is_file():
                log_event("base_charm_reuse", subject=subject, base_anat=str(base_anat), label=str(label_path))
                return base_anat
        except Exception as exc:
            log_event("base_charm_marker_invalid", subject=subject, marker=str(marker), error=str(exc))

    if base_root.exists():
        shutil.rmtree(base_root)

    inputs = resolve_source_inputs(source_root, subject)
    base_anat.mkdir(parents=True, exist_ok=True)
    copied_t1 = copy_input_file(inputs["t1"], base_anat)
    copied_t2 = copy_input_file(inputs["t2"], base_anat)
    copied_manual = copy_input_file(inputs["manual"], base_anat)

    manifest = {
        "subject": subject,
        "source_anat": str(inputs["anat"]),
        "source_t1": str(inputs["t1"]),
        "source_t2": str(inputs["t2"]),
        "source_manual_segmentation": str(inputs["manual"]),
        "copied_t1": str(copied_t1),
        "copied_t2": str(copied_t2),
        "copied_manual_segmentation": str(copied_manual),
    }
    write_json(base_root / "source_manifest.json", manifest)

    run_cmd(
        ["charm", subject, str(copied_t1), str(copied_t2), "--forcerun", "--forceqform"],
        cwd=base_anat,
        label="charm_init",
        deadline=deadline,
    )

    charm_label = find_charm_label(base_anat, subject)
    charm_original_copy = base_anat / f"{subject}_CHARM_original_tissue_labeling_upsampled.nii.gz"
    shutil.copy2(charm_label, charm_original_copy)
    mesh = find_mesh(base_anat, subject)

    write_json(
        marker,
        {
            "subject": subject,
            "base_anat": str(base_anat),
            "charm_label_path": str(charm_label),
            "charm_original_copy": str(charm_original_copy),
            "base_mesh_path": str(mesh) if mesh else None,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    log_event(
        "base_charm_complete",
        subject=subject,
        base_anat=str(base_anat),
        charm_label=str(charm_label),
        mesh=str(mesh) if mesh else None,
    )
    return base_anat


def to_int_labels(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(img.get_fdata(dtype=np.float32))
    return np.asarray(data, dtype=np.int16)


def resample_manual_to_charm(
    manual_img: nib.spatialimages.SpatialImage,
    charm_img: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    manual_int = nib.Nifti1Image(to_int_labels(manual_img), manual_img.affine, manual_img.header)
    manual_int.header.set_data_dtype(np.int16)
    if manual_img.shape == charm_img.shape and np.allclose(manual_img.affine, charm_img.affine, atol=1e-5):
        return nib.Nifti1Image(to_int_labels(manual_img), charm_img.affine, charm_img.header)
    resampled = resample_from_to(manual_int, charm_img, order=0)
    out = nib.Nifti1Image(to_int_labels(resampled), charm_img.affine, charm_img.header)
    out.header.set_data_dtype(np.int16)
    return out


def label_counts(data: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(data, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def component_summary(mask: np.ndarray) -> dict[str, Any]:
    from scipy import ndimage

    if not np.any(mask):
        return {"voxels": 0, "components": 0, "largest_components": []}
    labeled, n_components = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labeled.ravel())[1:]
    largest = sorted((int(x) for x in sizes), reverse=True)[:10]
    coords = np.argwhere(mask)
    return {
        "voxels": int(mask.sum()),
        "components": int(n_components),
        "largest_components": largest,
        "bbox_min": [int(x) for x in coords.min(axis=0)],
        "bbox_max": [int(x) for x in coords.max(axis=0)],
    }


def edge_voxel_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    edge = np.zeros(mask.shape, dtype=bool)
    edge[0, :, :] = True
    edge[-1, :, :] = True
    edge[:, 0, :] = True
    edge[:, -1, :] = True
    edge[:, :, 0] = True
    edge[:, :, -1] = True
    return int((mask & edge).sum())


def save_nifti(data: np.ndarray, like: nib.spatialimages.SpatialImage, path: Path, dtype: str = "int16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(dtype, copy=False), like.affine, like.header)
    out.header.set_data_dtype(dtype)
    nib.save(out, str(path))


def atomic_replace_nifti(src_path: Path, dst_path: Path, dtype: str = "uint16") -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.parent / f".tmp_{dst_path.name}"
    img = nib.load(str(src_path))
    data = np.asarray(img.dataobj).astype(dtype, copy=False)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.header.set_data_dtype(dtype)
    nib.save(out, str(tmp_path))
    with tmp_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_path, dst_path)


def merge_candidate_a(manual: np.ndarray, charm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    overlay = (manual != BACKGROUND_LABEL) & (manual != SKIN_LABEL)
    final = np.array(charm, copy=True)
    final[overlay] = manual[overlay]
    debug = {
        "strategy": "candidate_a",
        "base": "full_charm_tissue_map",
        "manual_overlay_rule": "manual label != 0 and manual label != 5",
        "manual_overlay_voxels": int(overlay.sum()),
        "manual_skin_voxels_ignored": int((manual == SKIN_LABEL).sum()),
        "manual_background_voxels_ignored": int((manual == BACKGROUND_LABEL).sum()),
        "charm_fallback_voxels": int((~overlay).sum()),
    }
    return final, debug


def merge_candidate_b(manual: np.ndarray, charm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    charm_head = charm != BACKGROUND_LABEL
    solid_head = ndimage.binary_fill_holes(charm_head)
    final = np.zeros(charm.shape, dtype=np.int16)
    final[solid_head] = SKIN_LABEL

    overlay = (manual != BACKGROUND_LABEL) & (manual != SKIN_LABEL)
    final[overlay] = manual[overlay]
    debug = {
        "strategy": "candidate_b",
        "base": "solid_binary_charm_head_envelope_labelled_as_skin",
        "manual_overlay_rule": "manual label != 0 and manual label != 5",
        "solid_head_voxels": int(solid_head.sum()),
        "manual_overlay_voxels": int(overlay.sum()),
        "manual_skin_voxels_ignored": int((manual == SKIN_LABEL).sum()),
        "manual_background_voxels_ignored": int((manual == BACKGROUND_LABEL).sum()),
        "remaining_base_skin_voxels": int((final == SKIN_LABEL).sum()),
    }
    return final, debug


def save_preview_png(
    *,
    manual: np.ndarray,
    charm: np.ndarray,
    final: np.ndarray,
    path: Path,
    title: str,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    head = final != BACKGROUND_LABEL
    if np.any(head):
        center = np.round(np.argwhere(head).mean(axis=0)).astype(int)
    else:
        center = np.array(final.shape) // 2

    arrays = [manual, charm, final]
    row_titles = ["Manual resampled", "CHARM original", "Merged final"]
    slice_specs = [
        ("sagittal", 0, int(center[0])),
        ("coronal", 1, int(center[1])),
        ("axial", 2, int(center[2])),
    ]
    vmax = max(int(np.max(manual)), int(np.max(charm)), int(np.max(final)), 1)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=12)
    for row, arr in enumerate(arrays):
        for col, (name, axis, idx) in enumerate(slice_specs):
            idx = max(0, min(idx, arr.shape[axis] - 1))
            if axis == 0:
                slc = arr[idx, :, :]
            elif axis == 1:
                slc = arr[:, idx, :]
            else:
                slc = arr[:, :, idx]
            axes[row, col].imshow(np.rot90(slc), interpolation="nearest", cmap="tab20", vmin=0, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"{name}={idx}")
            if col == 0:
                axes[row, col].set_ylabel(row_titles[row])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return None


def build_qc(
    *,
    subject: str,
    strategy: str,
    manual: np.ndarray,
    charm: np.ndarray,
    final: np.ndarray,
    debug: dict[str, Any],
    manual_img: nib.spatialimages.SpatialImage,
    charm_img: nib.spatialimages.SpatialImage,
) -> dict[str, Any]:
    manual_overlay = (manual != BACKGROUND_LABEL) & (manual != SKIN_LABEL)
    final_skin = final == SKIN_LABEL
    final_head = final != BACKGROUND_LABEL
    return {
        "subject": subject,
        "strategy": strategy,
        "merge_debug": debug,
        "manual_shape": list(manual_img.shape),
        "charm_shape": list(charm_img.shape),
        "output_shape": list(final.shape),
        "manual_affine": np.asarray(manual_img.affine).tolist(),
        "charm_affine": np.asarray(charm_img.affine).tolist(),
        "output_affine": np.asarray(charm_img.affine).tolist(),
        "label_counts": {
            "manual_resampled": label_counts(manual),
            "charm_original": label_counts(charm),
            "final": label_counts(final),
        },
        "components": {
            "final_skin": component_summary(final_skin),
            "final_head": component_summary(final_head),
        },
        "edge_voxels": {
            "manual_skin": edge_voxel_count(manual == SKIN_LABEL),
            "final_skin": edge_voxel_count(final_skin),
            "final_head": edge_voxel_count(final_head),
        },
        "manual_overlay_voxels_outside_charm_head": int((manual_overlay & (charm == BACKGROUND_LABEL)).sum()),
    }


def run_strategy(
    *,
    base_anat: Path,
    out_root: Path,
    subject: str,
    strategy: str,
    force: bool,
    deadline: float | None,
) -> None:
    strategy_info = STRATEGIES[strategy]
    subject_root = out_root / subject
    strategy_root = subject_root / strategy_info["directory"]
    strategy_anat = strategy_root / "anat"
    marker = strategy_marker(strategy_root)

    if force and strategy_root.exists():
        shutil.rmtree(strategy_root)

    if marker.is_file():
        try:
            payload = read_json(marker)
            mesh = Path(payload["mesh_path"])
            merged = Path(payload["merged_segmentation_path"])
            if mesh.is_file() and mesh.stat().st_size > 0 and merged.is_file():
                log_event("strategy_reuse", subject=subject, strategy=strategy, mesh=str(mesh))
                return
        except Exception as exc:
            log_event("strategy_marker_invalid", subject=subject, strategy=strategy, marker=str(marker), error=str(exc))

    if strategy_root.exists():
        shutil.rmtree(strategy_root)
    shutil.copytree(base_anat, strategy_anat)

    manual_path = first_existing(candidate_nifti_paths(strategy_anat, f"{subject}_T1w_ras_1mm_T1andT2_masks"))
    if manual_path is None:
        raise MissingInput(f"Manual segmentation missing from strategy workdir: {strategy_anat}")
    charm_label = find_charm_label(strategy_anat, subject)

    manual_img = nib.load(str(manual_path))
    charm_img = nib.load(str(charm_label))
    manual_resampled_img = resample_manual_to_charm(manual_img, charm_img)
    manual = to_int_labels(manual_resampled_img)
    charm = to_int_labels(charm_img)

    if strategy == "candidate_a":
        final, merge_debug = merge_candidate_a(manual, charm)
    elif strategy == "candidate_b":
        final, merge_debug = merge_candidate_b(manual, charm)
        from scipy import ndimage

        solid_head = ndimage.binary_fill_holes(charm != BACKGROUND_LABEL)
        save_nifti(
            solid_head.astype(np.uint8),
            charm_img,
            strategy_root / f"{subject}_{strategy}_solid_charm_head_mask.nii.gz",
            dtype="uint8",
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    manual_resampled_path = strategy_root / f"{subject}_manual_resampled_to_CHARM.nii.gz"
    merged_path = strategy_root / f"{subject}_{strategy}_merged.nii.gz"
    charm_original_path = strategy_root / f"{subject}_CHARM_original_tissue_labeling_upsampled.nii.gz"
    save_nifti(manual, charm_img, manual_resampled_path)
    shutil.copy2(charm_label, charm_original_path)
    save_nifti(final, charm_img, merged_path)
    atomic_replace_nifti(merged_path, charm_label, dtype="uint16")

    qc = build_qc(
        subject=subject,
        strategy=strategy,
        manual=manual,
        charm=charm,
        final=final,
        debug=merge_debug,
        manual_img=manual_img,
        charm_img=charm_img,
    )
    preview_error = save_preview_png(
        manual=manual,
        charm=charm,
        final=final,
        path=strategy_root / f"{subject}_{strategy}_preview.png",
        title=f"{subject} {strategy}: manual vs CHARM vs merged",
    )
    if preview_error:
        qc["preview_error"] = preview_error
    write_json(strategy_root / "merge_qc.json", qc)

    run_cmd(["charm", subject, "--mesh"], cwd=strategy_anat, label=f"{strategy}_charm_remesh", deadline=deadline)

    mesh = find_mesh(strategy_anat, subject)
    if mesh is None or mesh.stat().st_size == 0:
        raise RuntimeError(f"Remeshing did not produce a non-empty mesh for {subject} {strategy}")

    write_json(
        marker,
        {
            "subject": subject,
            "strategy": strategy,
            "description": strategy_info["description"],
            "strategy_anat": str(strategy_anat),
            "merged_segmentation_path": str(merged_path),
            "installed_charm_label_path": str(charm_label),
            "mesh_path": str(mesh),
            "qc_path": str(strategy_root / "merge_qc.json"),
            "preview_path": str(strategy_root / f"{subject}_{strategy}_preview.png"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    log_event("strategy_complete", subject=subject, strategy=strategy, mesh=str(mesh), merged=str(merged_path))


def selected_strategies(value: str) -> list[str]:
    if value == "both":
        return ["candidate_a", "candidate_b"]
    return [value]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Dataset root containing <subject>/anat, or a direct anat directory.")
    parser.add_argument("--out-root", required=True, help="Trial output root. Source data are copied here before CHARM runs.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. sub-CC620005.")
    parser.add_argument(
        "--strategies",
        choices=["both", "candidate_a", "candidate_b"],
        default="both",
        help="Merge strategies to run.",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=7.0,
        help="Wall-clock budget enforced inside the runner so Slurm can self-requeue before its time limit.",
    )
    parser.add_argument("--force", action="store_true", help="Remove this subject's trial outputs before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    subject = args.subject
    deadline = time.monotonic() + args.timeout_hours * 3600 if args.timeout_hours > 0 else None

    try:
        if args.force:
            subject_root = out_root / subject
            if subject_root.exists():
                shutil.rmtree(subject_root)

        base_anat = prepare_base_charm(
            source_root=source_root,
            out_root=out_root,
            subject=subject,
            force=False,
            deadline=deadline,
        )
        for strategy in selected_strategies(args.strategies):
            run_strategy(
                base_anat=base_anat,
                out_root=out_root,
                subject=subject,
                strategy=strategy,
                force=False,
                deadline=deadline,
            )
        log_event("trial_complete", subject=subject, out_root=str(out_root), strategies=selected_strategies(args.strategies))
        return 0
    except MissingInput as exc:
        log_event("missing_input", subject=subject, error=str(exc))
        return INPUT_EXIT_CODE
    except CommandTimeout as exc:
        log_event("trial_timeout", subject=subject, error=str(exc))
        return TIMEOUT_EXIT_CODE
    except Exception as exc:
        log_event("trial_error", subject=subject, error=str(exc), error_type=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
