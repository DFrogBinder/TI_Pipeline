from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import hashlib

from repeatability_combined_analysis_v2.settings import (
    CONDITIONS,
    OUTCOMES,
    SUBJECTS,
    TARGETS,
    VERSIONS,
    expected_participant_cells,
    expected_run_rows,
)
from repeatability_combined_analysis_v2.statistics import (
    build_matched_decomposition,
    summarize_participant_cells,
    summarize_under_anode,
    validate_final_matrix,
    validate_preliminary_matrix,
)


def final_matrix() -> pd.DataFrame:
    empty_provenance = "{}"
    empty_provenance_sha256 = hashlib.sha256(
        empty_provenance.encode("utf-8")
    ).hexdigest()
    rows = []
    for subject_index, subject in enumerate(SUBJECTS):
        for version_index, version in enumerate(VERSIONS):
            for target_index, target in enumerate(TARGETS):
                for condition_index, condition in enumerate(CONDITIONS):
                    for outcome_index, outcome in enumerate(OUTCOMES):
                        for repeat in range(1, 41):
                            base = (
                                0.1
                                + 0.01 * subject_index
                                + 0.02 * version_index
                                + 0.003 * target_index
                                + 0.004 * condition_index
                                + 0.005 * outcome_index
                            )
                            spread = (repeat - 20.5) * 1e-4 * (outcome_index + 1)
                            rows.append(
                                {
                                    "subject": subject,
                                    "version": version,
                                    "target": target,
                                    "condition": condition,
                                    "outcome_mode": outcome,
                                    "repeat": repeat,
                                    "repeat_tag": f"repeat_{repeat:03d}",
                                    "roi_mean_v_per_m": np.float64(base + spread + 0.001),
                                    "roi_median_v_per_m": np.float64(base + spread),
                                    "roi_p95_v_per_m": np.float64(base + spread + 0.01),
                                    "finite_volume_fraction": np.float64(1.0),
                                    "source_msh_path": f"/native/{version}/{subject}/{target}/{condition}/{outcome}/{repeat}.msh",
                                    "source_msh_sha256": "a" * 64,
                                    "source_field_name": (
                                        "TImax" if outcome == "ti_target" else "E"
                                    ),
                                    "geometry_sha256": (
                                        "d" * 64 if condition == "fixed_mesh" else "e" * 64
                                    ),
                                    "roi_provenance_sha256": empty_provenance_sha256,
                                    "roi_provenance_json": empty_provenance,
                                    "extraction_implementation": "repeatability_v2.native_mesh_extraction",
                                    "extraction_version": 2,
                                    "simnibs_version": (
                                        "3.2.6" if version == "simnibs326" else "4.6.0"
                                    ),
                                    "analysis_dtype": "float64",
                                    "stored_field_dtype": "float64",
                                    "preliminary": False,
                                }
                            )
    return pd.DataFrame(rows)


def test_final_scope_is_240_participant_cells_and_9600_run_rows() -> None:
    assert expected_participant_cells() == 240
    assert expected_run_rows() == 9600
    frame = final_matrix()
    validate_final_matrix(frame)
    assert len(frame) == 9600


def test_final_matrix_rejects_preliminary_401_and_float32() -> None:
    frame = final_matrix()
    frame.loc[0, "version"] = "simnibs401"
    frame.loc[0, "preliminary"] = True
    with pytest.raises(ValueError, match="4.0.1|preliminary"):
        validate_final_matrix(frame)

    frame = final_matrix()
    frame.loc[0, "analysis_dtype"] = "float32"
    with pytest.raises(ValueError, match="float64"):
        validate_final_matrix(frame)


def test_final_matrix_requires_complete_native_provenance() -> None:
    frame = final_matrix()
    frame.loc[0, "source_msh_sha256"] = ""

    with pytest.raises(ValueError, match="provenance"):
        validate_final_matrix(frame)


def test_final_matrix_rejects_fixed_geometry_drift() -> None:
    frame = final_matrix()
    fixed = frame.index[
        (frame["condition"] == "fixed_mesh")
        & (frame["subject"] == SUBJECTS[0])
        & (frame["version"] == VERSIONS[0])
        & (frame["target"] == TARGETS[0])
    ]
    frame.loc[fixed[-1], "geometry_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="Fixed geometry changed"):
        validate_final_matrix(frame)


def test_participant_cell_summaries_keep_repeats_nested() -> None:
    summary = summarize_participant_cells(final_matrix())

    assert len(summary) == 240 * 3
    assert set(summary["run_count"]) == {40}
    assert summary["subject"].nunique() == 10
    assert set(summary["metric"]) == {
        "roi_median_v_per_m",
        "roi_mean_v_per_m",
        "roi_p95_v_per_m",
    }
    assert {"cell_sd", "cell_cv", "cell_median"}.issubset(summary.columns)


def test_matched_decomposition_is_participant_paired_and_excludes_under_anode() -> None:
    summary = summarize_participant_cells(final_matrix())
    decomposition = build_matched_decomposition(summary)

    assert decomposition["participant_count"].eq(10).all()
    assert set(decomposition["comparison"]) == {
        "tdcs_pair1_matched_target_minus_ti_target"
    }
    assert not decomposition.astype(str).apply(
        lambda column: column.str.contains("under_anode").any()
    ).any()
    assert decomposition["interpretation"].eq(
        "matched_roi_ti_component_variability_decomposition"
    ).all()
    assert decomposition["assumes_ti_doubles_variability"].eq(False).all()


def test_under_anode_analysis_is_separate_application_facing_output() -> None:
    summary = summarize_participant_cells(final_matrix())
    under = summarize_under_anode(summary)

    assert set(under["outcome_mode"]) == {"tdcs_pair1_under_anode"}
    assert set(under["participant_count"]) == {10}
    assert under["interpretation"].eq("separate_application_facing_tdcs").all()
    assert under["location_matched_to_ti"].eq(False).all()


def preliminary_matrix() -> pd.DataFrame:
    frame = final_matrix()
    frame.loc[frame["version"] == "simnibs46", "version"] = "simnibs401"
    frame.loc[frame["version"] == "simnibs401", "simnibs_version"] = "4.0.1"
    frame["preliminary"] = True
    return frame


def test_preliminary_matrix_accepts_exact_326_401_scope_only() -> None:
    frame = preliminary_matrix()

    validate_preliminary_matrix(frame)
    summary = summarize_participant_cells(frame, final=False)
    decomposition = build_matched_decomposition(
        summary, versions=("simnibs326", "simnibs401")
    )

    assert len(frame) == 9600
    assert len(summary) == 240 * 3
    assert set(decomposition["version"]) == {"simnibs326", "simnibs401"}

    frame.loc[0, "preliminary"] = False
    with pytest.raises(ValueError, match="preliminary"):
        validate_preliminary_matrix(frame)
