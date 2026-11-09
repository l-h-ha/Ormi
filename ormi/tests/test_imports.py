import importlib
import os
import subprocess
import sys
from importlib.metadata import requires, version

import pytest

import ormi as om
import ormi_core as core

###
###
###

@pytest.mark.parametrize("name", core.__all__)
def test_public_exports_are_the_core_objects(name):
    assert getattr(om, name) is getattr(core, name)

def test_exports_and_installed_version_match_the_core():
    assert om.__all__ == [*core.__all__, "save", "load", "batches", "fit"]
    assert not {"save", "load", "batches", "fit"}.intersection(core.__all__)
    assert om.__version__ == core.__version__ == version("ormi")

def test_training_progress_dependency_is_installed_with_ormi():
    assert any(requirement.startswith("tqdm") and "extra ==" not in requirement
               for requirement in requires("ormi") or [])

@pytest.mark.parametrize("name", ["nn", "ops", "optim", "jit"])
def test_qualified_imports_return_the_existing_modules(name):
    module = importlib.import_module("ormi." + name)
    assert module is getattr(om, name) is getattr(core, name)

def test_from_imports_use_the_core_objects():
    from ormi.jit import clear
    from ormi.nn import Linear, Module
    from ormi.ops import matmul
    from ormi.optim import sgd

    assert Linear is core.nn.Linear
    assert Module is core.nn.Module
    assert matmul is core.ops.matmul
    assert sgd is core.sgd
    assert clear is core.jit.clear

def test_import_and_graph_construction_do_not_require_a_compiler():
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "ormi-compiler-does-not-exist"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ormi as om; from ormi.nn import Linear; "
            "x = om.tensor([1., 2.]); y = Linear(2, 3, seed=7)(x).sin(); "
            "dx = om.grad(y.sum(), x); "
            "dxx = om.partial(y, x, coordinate=0, component=0, order=2); "
            "assert not y.realized and not dx.realized and not dxx.realized",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
