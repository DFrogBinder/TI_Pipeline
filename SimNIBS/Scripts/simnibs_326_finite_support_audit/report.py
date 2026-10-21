"""Human-readable, evidence-bounded report for collected audit results."""

from __future__ import annotations

from typing import Any


def _count(mapping: dict[str, int], key: str) -> int:
    return int(mapping.get(key, 0))


def build_report(summary: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    v3 = summary["v3_cause_counts"]
    logs = summary["production_log_status_counts"]
    direct_mesh = _count(v3, "mesh_field_nonfinite")
    interpolation = _count(v3, "interpolation_nonfinite") + _count(
        v3, "mixed_interpolation_and_mask_exclusion"
    )
    mask = _count(v3, "brain_mask_exclusion") + _count(
        v3, "mixed_interpolation_and_mask_exclusion"
    )
    inconsistent = _count(v3, "masked_output_inconsistent")
    unproven_logs = sum(
        int(value)
        for key, value in logs.items()
        if key != "successful_final_attempt"
    )
    lines = [
        "# SimNIBS v3 finite-support forensic audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope and interpretation boundary",
        "",
        (
            f"The audit inspected {summary['v3_runs_audited']:,} retained SimNIBS 3.2.6 "
            f"runs, {summary['matched_v4_runs_audited']:,} matched SimNIBS 4.0.1 outputs, "
            f"and {summary['direct_v3_mesh_fields_audited']:,} direct v3 TImax mesh fields."
        ),
        "",
        (
            "Support means finite numerical values inside the optimizer-matched, "
            "parcel-clipped spherical ROI. It is not the fraction above 0.2 V/m, and "
            "0.2 V/m is not used anywhere in this audit."
        ),
        "",
        "## Stage-localization result",
        "",
        f"- Direct v3 ROI mesh field already non-finite: **{direct_mesh:,} runs**.",
        f"- Loss present in the raw interpolated field: **{interpolation:,} runs**.",
        f"- Finite raw values excluded by the WM/GM mask: **{mask:,} runs**.",
        f"- Final masked field inconsistent with raw field plus tissue mask: **{inconsistent:,} runs**.",
        f"- Production histories without a proven successful final attempt: **{unproven_logs:,} runs**.",
        "",
        f"Full v3 cause counts: `{v3}`.",
        "",
        f"Matched v4 cause counts: `{summary['v4_cause_counts']}`.",
        "",
        "## Subject-target-condition summaries",
        "",
        (
            "| Target | Subject | Condition | v3 finite fraction, mean [min, max] | "
            "v4 finite fraction, mean [min, max] | v4-v3 mean | v3 cause counts |"
        ),
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in groups:
        lines.append(
            "| {target} | {subject} | {condition} | {v3mean:.4f} [{v3min:.4f}, {v3max:.4f}] "
            "| {v4mean:.4f} [{v4min:.4f}, {v4max:.4f}] | {delta:.4f} | `{causes}` |".format(
                target=row["target"],
                subject=row["subject"],
                condition=row["condition"],
                v3mean=float(row["v3_finite_roi_fraction_mean"]),
                v3min=float(row["v3_finite_roi_fraction_min"]),
                v3max=float(row["v3_finite_roi_fraction_max"]),
                v4mean=float(row["v4_finite_roi_fraction_mean"]),
                v4min=float(row["v4_finite_roi_fraction_min"]),
                v4max=float(row["v4_finite_roi_fraction_max"]),
                delta=float(row["v4_minus_v3_fraction_mean"]),
                causes=row["v3_cause_counts_json"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            "`all_runs.csv` contains the run-level stage counts and paths; "
            "`audit_summary.csv` contains the table above; `cells/*/*/figures` contains "
            "the 40 matched panels; and `manifest.json` plus `checksums.sha256` records provenance.",
            "",
        ]
    )
    return "\n".join(lines)
