import os
from importlib.metadata import version
from pathlib import Path

import ormi_core as oc


def _cuda_plugin() -> Path:
    suffix = ".dll" if os.name == "nt" else ".so"
    return Path(oc.__file__).parent / "_ext" / f"{'lib' if suffix == '.so' else ''}ormi_cuda_backend{suffix}"


def test_core_install_is_cpu_only():
    assert not _cuda_plugin().exists()


def test_cuda_install_matches_core():
    assert version("ormi-cuda") == version("ormi-core")
    assert _cuda_plugin().is_file()

    result = (oc.Tensor([1.0, 2.0]) + 1.0).numpy()
    assert result.tolist() == [2.0, 3.0]
