from pathlib import Path

from post.utils.collect_post_outputs import (
    build_copy_manifest,
    build_export_plan,
    execute_export_plan,
)


def test_collect_post_outputs_slurm_prefers_activated_conda_python():
    script = (
        Path(__file__).resolve().parents[1]
        / "HPC_scripts"
        / "collect_post_outputs.slurm"
    )
    text = script.read_text(encoding="utf-8")

    assert '${CONDA_PREFIX}/bin/python' in text
    assert '"${PYTHON}" != /*' in text
    assert 'hash -r' in text
    expected_repeats = " ".join(f"{idx:02d}" for idx in range(1, 41))
    assert f'POST_COLLECT_REPEATS="${{POST_COLLECT_REPEATS:-{expected_repeats}}}"' in text


def test_collect_post_outputs_slurm_prefers_stanage_checkout_over_spool_path():
    script = (
        Path(__file__).resolve().parents[1]
        / "HPC_scripts"
        / "collect_post_outputs.slurm"
    )
    text = script.read_text(encoding="utf-8")

    assert 'STANAGE_REPO_DIR="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment"' in text
    assert 'REPO_DIR="${STANAGE_REPO_DIR}"' in text
    assert 'REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text


def test_build_export_plan_for_batch_root_includes_post_population_and_summary(tmp_path):
    dataset_root = tmp_path / "Left_Hippocampus_Data_01"
    post_dir = dataset_root / "sub-01" / "anat" / "post"
    population_dir = dataset_root / "population_analysis"
    summary_file = tmp_path / "post_processing_batch_summary.json"

    post_dir.mkdir(parents=True)
    population_dir.mkdir(parents=True)
    (post_dir / "subject_metrics.json").write_text("{}", encoding="utf-8")
    (population_dir / "subject_robustness.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    summary_file.write_text("{}", encoding="utf-8")

    plan = build_export_plan(root=tmp_path)

    assert [item.rel_path.as_posix() for item in plan] == [
        "Left_Hippocampus_Data_01/population_analysis",
        "Left_Hippocampus_Data_01/sub-01/anat/post",
        "post_processing_batch_summary.json",
    ]


def test_build_export_plan_falls_back_to_single_dataset_root(tmp_path):
    post_dir = tmp_path / "sub-01" / "anat" / "post"
    post_dir.mkdir(parents=True)
    (post_dir / "region_stats_fastsurfer.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    plan = build_export_plan(root=tmp_path, summary_glob=None)

    assert [item.rel_path.as_posix() for item in plan] == ["sub-01/anat/post"]


def test_execute_export_plan_copies_full_directory_tree_and_summary_file(tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    dataset_root = src_root / "Left_Hippocampus_Data_01"
    post_dir = dataset_root / "sub-01" / "anat" / "post"
    summary_file = src_root / "post_processing_batch_summary.json"

    post_dir.mkdir(parents=True)
    (post_dir / "subject_metrics.json").write_text('{"ok": true}', encoding="utf-8")
    (post_dir / "nested").mkdir()
    (post_dir / "empty_dir").mkdir()
    (post_dir / "nested" / "figure.png").write_text("png", encoding="utf-8")
    summary_file.write_text('{"datasets": 1}', encoding="utf-8")

    plan = build_export_plan(root=src_root, include_population=False)
    execute_export_plan(plan, dest=dest_root, dry_run=False)

    assert (dest_root / "Left_Hippocampus_Data_01" / "sub-01" / "anat" / "post" / "subject_metrics.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert (dest_root / "Left_Hippocampus_Data_01" / "sub-01" / "anat" / "post" / "nested" / "figure.png").read_text(encoding="utf-8") == "png"
    assert (dest_root / "Left_Hippocampus_Data_01" / "sub-01" / "anat" / "post" / "empty_dir").is_dir()
    assert (dest_root / "post_processing_batch_summary.json").read_text(encoding="utf-8") == '{"datasets": 1}'


def test_build_copy_manifest_expands_directories_to_file_level_operations(tmp_path):
    src_root = tmp_path / "src"
    dest_root = tmp_path / "dest"
    dataset_root = src_root / "Left_Hippocampus_Data_01"
    post_dir = dataset_root / "sub-01" / "anat" / "post"
    summary_file = src_root / "post_processing_batch_summary.json"

    post_dir.mkdir(parents=True)
    (post_dir / "a.txt").write_text("a", encoding="utf-8")
    (post_dir / "nested").mkdir()
    (post_dir / "nested" / "b.txt").write_text("b", encoding="utf-8")
    summary_file.write_text("{}", encoding="utf-8")

    plan = build_export_plan(root=src_root, include_population=False)
    directories, file_copies = build_copy_manifest(plan, dest=dest_root)

    assert sorted(path.relative_to(dest_root).as_posix() for path in directories) == [
        ".",
        "Left_Hippocampus_Data_01/sub-01/anat/post",
        "Left_Hippocampus_Data_01/sub-01/anat/post/nested",
    ]
    assert [(src.relative_to(src_root).as_posix(), dst.relative_to(dest_root).as_posix()) for src, dst in file_copies] == [
        (
            "Left_Hippocampus_Data_01/sub-01/anat/post/a.txt",
            "Left_Hippocampus_Data_01/sub-01/anat/post/a.txt",
        ),
        (
            "Left_Hippocampus_Data_01/sub-01/anat/post/nested/b.txt",
            "Left_Hippocampus_Data_01/sub-01/anat/post/nested/b.txt",
        ),
        (
            "post_processing_batch_summary.json",
            "post_processing_batch_summary.json",
        ),
    ]
