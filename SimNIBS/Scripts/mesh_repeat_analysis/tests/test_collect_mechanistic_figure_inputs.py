import builtins
import sys
from unittest import mock

import pytest

from post import collect_mechanistic_figure_inputs as collector


def test_nifti_dependency_error_names_missing_package_and_hpc_fixes():
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nibabel":
            raise ModuleNotFoundError("No module named 'nibabel'", name="nibabel")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", fake_import):
        with pytest.raises(RuntimeError) as exc_info:
            collector._import_nifti_dependencies()

    message = str(exc_info.value)
    assert "Missing Python package(s): nibabel" in message
    assert "module load SimNIBS/4.0.1-foss-2023a" in message
    assert "--coverage-csv" in message


def test_dependency_check_cli_does_not_require_collection_paths(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["collect_mechanistic_figure_inputs.py", "--check-nifti-dependencies"])
    monkeypatch.setattr(collector, "_import_nifti_dependencies", lambda: (object(), object()))

    collector.main()

    assert "nifti_dependencies=ok" in capsys.readouterr().out
