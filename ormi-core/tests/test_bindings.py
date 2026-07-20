from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

import ormi_core as oc


def test_alpha_version_and_generated_submodules():
    assert oc.__version__ == "0.1.0a4"
    assert callable(oc.ops.matmul)
    assert callable(oc.optim.sgd)
    assert callable(oc.jit.info)


def test_exception_hierarchy_is_public():
    with pytest.raises(oc.DeviceError):
        oc.Device("not-a-device")
    assert issubclass(oc.DeviceError, oc.OrmiError)
    assert issubclass(oc.cudaError, oc.DeviceError)


def test_generated_stub_modules_are_valid_python():
    stub_dir = Path(oc.__file__).parent / "_ext" / "_C"
    expected_stubs = {"__init__.pyi", "ops.pyi", "optim.pyi", "jit.pyi"}
    expected_sources = {"ops.py", "optim.py", "jit.py"}

    assert {path.name for path in stub_dir.glob("*.pyi")} == expected_stubs
    assert {path.name for path in stub_dir.glob("*.py")} == expected_sources
    for path in stub_dir.glob("*.pyi"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_empty_jit_cache_is_inspectable():
    info = oc.jit.info()
    assert info.entries >= 0
    assert info.bytes >= 0
    assert info.limit >= 0


def test_release_wheel_contains_cuda_plugin_when_requested():
    if os.environ.get("ORMI_WHEEL_EXPECT_CUDA") != "1":
        pytest.skip("CUDA plugin is optional outside release-wheel builds")

    extension_dir = Path(oc.__file__).parent / "_ext"
    names = {path.name for path in extension_dir.iterdir()}
    assert names & {"ormi_cuda_backend.dll", "libormi_cuda_backend.so"}
