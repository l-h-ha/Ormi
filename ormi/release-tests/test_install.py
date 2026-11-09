import importlib.util
import os
from importlib.metadata import requires, version
from pathlib import Path
import subprocess
import sys

import numpy as np
from packaging.requirements import Requirement

import ormi as om
import ormi_core as core

###
###
###

def test_versions_dependencies_and_install_locations():
    expected = os.environ["ORMI_RELEASE_VERSION"]
    assert om.__version__ == core.__version__ == version("ormi") == version("ormi-core") == expected

    for module in (om, core):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

    dependencies = {requirement.name: requirement for requirement in map(Requirement, requires("ormi"))}
    assert str(dependencies["ormi-core"].specifier) == f"=={expected}"
    assert str(dependencies["ormi-cuda"].specifier) == f"=={expected}"
    assert dependencies["ormi-cuda"].marker.evaluate({"extra": "cuda"})
    assert not dependencies["ormi-cuda"].marker.evaluate({"extra": ""})
    assert "tqdm" in dependencies and version("tqdm")
    assert importlib.util.find_spec("ormi_cuda") is None

def test_core_objects_and_qualified_imports_are_shared():
    for name in core.__all__:
        assert getattr(om, name) is getattr(core, name)

    for name in ("nn", "ops", "optim", "jit"):
        assert importlib.import_module(f"ormi.{name}") is getattr(core, name)

def test_graph_building_without_a_compiler():
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "ormi-compiler-does-not-exist"
    script = (
        "import ormi as om; "
        "x = om.tensor([1., 2.]); y = (x * x * x).sum(); "
        "dx = om.grad(y, x); dxx = om.grad(dx.sum(), x); "
        "assert not y.realized and not dx.realized and not dxx.realized"
    )
    subprocess.run([sys.executable, "-I", "-c", script], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)

def test_lazy_higher_derivatives_and_reused_plan():
    values = np.arange(12, dtype=np.float64).reshape(2, 1, 3, 2) / 10
    x = om.tensor(values, om.float64)
    output = (x * x * x).sum()
    first = om.grad(output, x)
    second = om.grad(first.sum(), x)
    third = om.grad(second.sum(), x)
    assert all(not value.realized for value in (output, first, second, third))
    plan = om.Plan([output, first, second, third])

    for data in (values, values + 0.25):
        x.copy_from(data)
        plan.run()
        np.testing.assert_allclose(output.item(), (data**3).sum(), rtol=1e-12)
        np.testing.assert_allclose(first.numpy(), 3 * data**2, rtol=1e-12)
        np.testing.assert_allclose(second.numpy(), 6 * data, rtol=1e-12)
        np.testing.assert_allclose(third.numpy(), np.full_like(data, 6), rtol=1e-12)

def test_training_and_checkpoint_restore_existing_tensors(tmp_path):
    values = np.sin(np.arange(42, dtype=np.float64)).reshape(7, 1, 3, 2)
    targets = 0.5 * values[..., :1] - 0.2 * values[..., 1:] + 0.1
    model = om.nn.Linear(2, 1, dtype=om.float64, seed=7)
    parameters = model.parameters()
    initial_state = [parameter.numpy() for parameter in parameters]
    probe = om.tensor(values[:2], om.float64)
    prediction = model(probe)
    plan = om.Plan([prediction])
    plan.run()
    initial_output = prediction.numpy()
    initial = tmp_path / "initial.npz"
    trained = tmp_path / "trained.npz"
    om.save(model, initial)

    history = om.fit(
        model, values, targets, loss=lambda x, y: om.nn.mse_loss(model(x), y),
        update=lambda parameter, gradient: om.sgd(parameter, gradient, 0.05),
        epochs=6, batch_size=3, seed=11, verbose=0,
    )
    assert set(history) == set(range(1, 7))
    assert history[6] < history[1]
    plan.run()
    trained_output = prediction.numpy()
    assert not np.allclose(trained_output, initial_output)
    om.save(model, trained)
    om.load(model, initial)

    for current, original, expected in zip(model.parameters(), parameters, initial_state, strict=True):
        assert current is original
        np.testing.assert_array_equal(current.numpy(), expected)

    plan.run()
    np.testing.assert_allclose(prediction.numpy(), initial_output, rtol=1e-12)
    om.load(model, trained)
    plan.run()
    np.testing.assert_allclose(prediction.numpy(), trained_output, rtol=1e-12)

def test_training_with_interacting_point_higher_derivatives():
    class Cubic(om.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = om.tensor(1.5, om.float64)
            self.register_parameter(self.weight)

        def forward(self, points):
            centered = points - points.mean((0, 1, 2), keepdims=True)
            return self.weight * self.weight * centered * centered * centered

    model = Cubic()
    values = np.sin(np.arange(15)).reshape(5, 1, 3, 1)
    targets = np.cos(np.arange(15)).reshape(5, 1, 3, 1)

    def loss(points, labels):
        second = om.partial(model(points), points, order=2)
        assert not second.realized
        return om.nn.mse_loss(second, labels)

    history = om.fit(
        model, values, targets, loss=loss, update=lambda p, g: om.sgd(p, g, 1e-4),
        epochs=2, batch_size=2, shuffle=False, verbose=0,
    )
    weight = 1.5

    for epoch in range(1, 3):
        total = 0.0

        for x, y in om.batches(values, targets, batch_size=2):
            coefficient = 6 * (x - x.mean()) * (1 - 1 / x.size)**2
            predicted = coefficient * weight**2
            total += len(x) * np.mean((predicted - y)**2)
            gradient = np.mean(4 * weight * coefficient * (predicted - y))
            weight -= 1e-4 * gradient

        np.testing.assert_allclose(history[epoch], total / len(values), rtol=1e-12)

    np.testing.assert_allclose(model.weight.item(), weight, rtol=1e-12)
