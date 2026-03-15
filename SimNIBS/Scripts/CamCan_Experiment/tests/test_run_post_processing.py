import pytest

from post.run_post_processing import PostBatchConfig, resolve_max_workers


def test_resolve_max_workers_uses_slurm_cpus_per_task(monkeypatch):
    monkeypatch.delenv("POST_MAX_WORKERS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    assert resolve_max_workers(cfg, task_count=10) == 6


def test_resolve_max_workers_prefers_explicit_post_override(monkeypatch):
    monkeypatch.setenv("POST_MAX_WORKERS", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    assert resolve_max_workers(cfg, task_count=10) == 3


def test_resolve_max_workers_rejects_invalid_env_values(monkeypatch):
    monkeypatch.setenv("POST_MAX_WORKERS", "0")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    with pytest.raises(ValueError, match="POST_MAX_WORKERS must be at least 1"):
        resolve_max_workers(cfg, task_count=10)
