"""Immutable final scope for native-mesh repeatability analysis v2."""

SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC510415",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)
VERSIONS = ("simnibs326", "simnibs46")
TARGETS = ("left_hippocampus", "right_m1")
CONDITIONS = ("remesh", "fixed_mesh")
OUTCOMES = (
    "ti_target",
    "tdcs_pair1_matched_target",
    "tdcs_pair1_under_anode",
)
METRICS = (
    "roi_median_v_per_m",
    "roi_mean_v_per_m",
    "roi_p95_v_per_m",
)
PRIMARY_METRIC = "roi_median_v_per_m"
REPEAT_COUNT = 40


def expected_participant_cells() -> int:
    return (
        len(SUBJECTS)
        * len(VERSIONS)
        * len(TARGETS)
        * len(CONDITIONS)
        * len(OUTCOMES)
    )


def expected_run_rows() -> int:
    return expected_participant_cells() * REPEAT_COUNT

