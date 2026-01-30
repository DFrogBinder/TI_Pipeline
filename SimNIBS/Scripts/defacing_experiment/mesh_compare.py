#!/usr/bin/env python3
# mesh_compare.py
# Compare two .msh files: counts, labels, bbox, and point/cell differences.

import argparse
from collections import Counter
import json
import sys

import numpy as np

try:
    import meshio
except ImportError as e:
    print("ERROR: meshio is required. Install with: pip install meshio", file=sys.stderr)
    raise

try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_mesh(path):
    mesh = meshio.read(path)
    return mesh


def bbox(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return mn, mx


def canonicalize_points(points, tol):
    if tol <= 0:
        # Exact comparison (no rounding)
        rounded = points
    else:
        rounded = np.round(points / tol).astype(np.int64)
    # Use structured array to compare rows regardless of order
    dtype = np.dtype([("x", rounded.dtype), ("y", rounded.dtype), ("z", rounded.dtype)])
    structured = rounded.view(dtype).reshape(-1)
    return structured


def multiset_counts(structured):
    uniq, counts = np.unique(structured, return_counts=True)
    return Counter(dict(zip(uniq.tolist(), counts.tolist())))


def cell_stats(mesh):
    stats = {}
    total_cells = 0
    for block in mesh.cells:
        stats[block.type] = stats.get(block.type, 0) + len(block.data)
        total_cells += len(block.data)
    return total_cells, stats


def label_stats(mesh):
    # Collect per cell block label distributions, for all cell_data keys.
    # mesh.cell_data is dict: key -> list (one per cell block)
    results = {}
    for key, data_list in mesh.cell_data.items():
        per_type = {}
        for block, data in zip(mesh.cells, data_list):
            # data can be (n,) or (n,1)
            arr = np.array(data).reshape(-1)
            per_type.setdefault(block.type, Counter())
            per_type[block.type].update(arr.tolist())
        results[key] = per_type
    return results


def compare_counters(a, b):
    all_keys = set(a.keys()) | set(b.keys())
    diffs = {}
    for k in all_keys:
        av = a.get(k, 0)
        bv = b.get(k, 0)
        if av != bv:
            diffs[k] = (av, bv)
    return diffs


def compare_label_stats(a, b):
    diffs = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        per_type_a = a.get(key, {})
        per_type_b = b.get(key, {})
        type_keys = set(per_type_a.keys()) | set(per_type_b.keys())
        for t in type_keys:
            ca = per_type_a.get(t, Counter())
            cb = per_type_b.get(t, Counter())
            d = compare_counters(ca, cb)
            if d:
                diffs.setdefault(key, {})[t] = d
    return diffs


def summarize_mesh(mesh, name, tol, quiet=False):
    pts = mesh.points
    mn, mx = bbox(pts)
    size = mx - mn
    vol = float(np.prod(size))
    total_cells, cell_counts = cell_stats(mesh)
    labels = label_stats(mesh)

    if quiet:
        return {
            "points": len(pts),
            "bbox_min": mn.tolist(),
            "bbox_max": mx.tolist(),
            "bbox_size": size.tolist(),
            "bbox_vol": vol,
            "total_cells": total_cells,
            "cells_by_type": dict(sorted(cell_counts.items())),
            "labels": labels,
        }

    print(f"== {name} ==")
    print(f"Tolerance: {tol}")
    print(f"Points: {len(pts)}")
    print(f"Bounding box min: {mn}")
    print(f"Bounding box max: {mx}")
    print(f"Bounding box size: {size}")
    print(f"Bounding box volume: {vol:.6g}")
    print(f"Total cells: {total_cells}")
    print("Cells by type:")
    for k, v in sorted(cell_counts.items()):
        print(f"  {k}: {v}")
    if labels:
        print("Label stats by cell_data key:")
        for key, per_type in labels.items():
            print(f"  key: {key}")
            for t, counter in per_type.items():
                top = counter.most_common(5)
                print(f"    {t}: {len(counter)} labels (top5: {top})")
    else:
        print("No cell_data labels found.")
    print()

    return None


def rms_and_max_nn_dist(a, b):
    if not _HAS_SCIPY:
        return None
    tree = cKDTree(b)
    dists, _ = tree.query(a, k=1)
    rms = float(np.sqrt(np.mean(dists ** 2)))
    mx = float(np.max(dists))
    return rms, mx


def main():
    ap = argparse.ArgumentParser(description="Compare two .msh meshes")
    ap.add_argument("mesh_a", help="first .msh file")
    ap.add_argument("mesh_b", help="second .msh file")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="point comparison tolerance (default: 1e-6)",
    )
    ap.add_argument(
        "--report-all-labels",
        action="store_true",
        help="print full per-label distributions (not just top5)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="print only a final verdict (useful for CI)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON report",
    )
    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(2)
    args = ap.parse_args()

    mesh_a = load_mesh(args.mesh_a)
    mesh_b = load_mesh(args.mesh_b)

    summary_a = summarize_mesh(mesh_a, "Mesh A", args.tol, quiet=args.quiet or args.json)
    summary_b = summarize_mesh(mesh_b, "Mesh B", args.tol, quiet=args.quiet or args.json)

    if not (args.quiet or args.json):
        print("== Differences ==")
    # Points count / bbox
    pts_a = mesh_a.points
    pts_b = mesh_b.points
    point_diff = False
    if len(pts_a) != len(pts_b):
        point_diff = True
        if not (args.quiet or args.json):
            print(f"Point count differs: {len(pts_a)} vs {len(pts_b)} (Δ {len(pts_b) - len(pts_a)})")
    else:
        # Compare sets of points (order independent)
        ca = multiset_counts(canonicalize_points(pts_a, args.tol))
        cb = multiset_counts(canonicalize_points(pts_b, args.tol))
        if ca == cb:
            if not (args.quiet or args.json):
                print("Point sets match (within tolerance).")
            if not (args.quiet or args.json) and _HAS_SCIPY:
                rms_max = rms_and_max_nn_dist(pts_a, pts_b)
                if rms_max is not None:
                    rms, mx = rms_max
                    print(f"Nearest-neighbor distance A→B: RMS {rms:.6g}, max {mx:.6g}")
        else:
            point_diff = True
            diff = compare_counters(ca, cb)
            if not (args.quiet or args.json):
                print(f"Point sets differ (within tolerance). Diff entries: {len(diff)}")
                # show a few
                for i, (k, v) in enumerate(diff.items()):
                    if i >= 5:
                        break
                    print(f"  point {k}: {v[0]} vs {v[1]}")

    # Cells
    total_a, cells_a = cell_stats(mesh_a)
    total_b, cells_b = cell_stats(mesh_b)
    cell_diff = False
    if total_a != total_b:
        cell_diff = True
        if not (args.quiet or args.json):
            print(f"Total cell count differs: {total_a} vs {total_b} (Δ {total_b - total_a})")
    cell_diffs = compare_counters(cells_a, cells_b)
    if cell_diffs:
        cell_diff = True
        if not (args.quiet or args.json):
            print("Cell count differences by type:")
            for k, (av, bv) in sorted(cell_diffs.items()):
                print(f"  {k}: {av} vs {bv} (Δ {bv - av})")
    else:
        if not (args.quiet or args.json):
            print("Cell counts by type match.")

    # Labels
    labels_a = label_stats(mesh_a)
    labels_b = label_stats(mesh_b)
    label_diffs = compare_label_stats(labels_a, labels_b)
    label_diff = False
    if label_diffs:
        label_diff = True
        if not (args.quiet or args.json):
            print("Label differences:")
            for key, per_type in label_diffs.items():
                print(f"  key: {key}")
                for t, diff in per_type.items():
                    # sort by absolute delta descending
                    items = sorted(diff.items(), key=lambda kv: abs(kv[1][1] - kv[1][0]), reverse=True)
                    print(f"    {t}: {len(diff)} differing labels")
                    # show a few label diffs
                    for i, (lab, (av, bv)) in enumerate(items):
                        if i >= 5:
                            break
                        print(f"      label {lab}: {av} vs {bv} (Δ {bv - av})")
    else:
        if not (args.quiet or args.json):
            print("Label distributions match.")

    if not (args.quiet or args.json) and args.report_all_labels:
        print("\n== Full label distributions (Mesh A vs Mesh B) ==")
        all_keys = set(labels_a.keys()) | set(labels_b.keys())
        for key in all_keys:
            print(f"key: {key}")
            per_type_a = labels_a.get(key, {})
            per_type_b = labels_b.get(key, {})
            for t in sorted(set(per_type_a) | set(per_type_b)):
                print(f"  {t}:")
                print(f"    A: {per_type_a.get(t, Counter())}")
                print(f"    B: {per_type_b.get(t, Counter())}")

    verdict_parts = []
    if point_diff:
        verdict_parts.append("points")
    if cell_diff:
        verdict_parts.append("cells")
    if label_diff:
        verdict_parts.append("labels")
    verdict = "MATCH" if not verdict_parts else f"DIFF: {', '.join(verdict_parts)}"

    if args.json:
        out = {
            "verdict": verdict,
            "tolerance": args.tol,
            "point_diff": point_diff,
            "cell_diff": cell_diff,
            "label_diff": label_diff,
            "mesh_a": summary_a,
            "mesh_b": summary_b,
            "cell_type_diffs": cell_diffs,
            "label_diffs": label_diffs,
        }
        if _HAS_SCIPY and len(pts_a) == len(pts_b):
            rms_max = rms_and_max_nn_dist(pts_a, pts_b)
            if rms_max is not None:
                out["nn_dist_a_to_b_rms"] = rms_max[0]
                out["nn_dist_a_to_b_max"] = rms_max[1]
        print(json.dumps(out, indent=2))
        return

    if args.quiet:
        print(verdict)
    else:
        print(f"Verdict: {verdict}")
        if not _HAS_SCIPY:
            print("Note: scipy not available; NN distance stats skipped.")


if __name__ == "__main__":
    main()
