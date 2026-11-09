import argparse
import runpy
from pathlib import Path

import numpy as np
import pytest

import ormi as om
from ormi import nn

###
###
###

@pytest.fixture(scope="module")
def pinn_example():
    path = Path(__file__).parents[1] / "examples" / "pinn.py"
    return runpy.run_path(str(path))

###
###
###

@pytest.mark.parametrize("width", [1, 7])
def test_example_uses_seeded_nn_layers(pinn_example, api_device, width):
    first = pinn_example["init_model"](width, api_device, 7)
    repeated = pinn_example["init_model"](width, api_device, 7)
    changed = pinn_example["init_model"](width, api_device, 8)
    layers = first.children()
    assert [type(layer) for layer in layers] == [nn.Linear, nn.Sin, nn.Linear]
    assert layers[0].in_features == 1
    assert layers[0].out_features == width
    assert layers[2].in_features == width
    assert layers[2].out_features == 1
    assert first.parameters() == [
        layers[0].weight, layers[0].bias, layers[2].weight, layers[2].bias
    ]

    for parameter, again in zip(first.parameters(), repeated.parameters(), strict=True):
        assert parameter.realized and parameter.op == "leaf"
        assert parameter.dtype == om.float64
        assert parameter.device == om.Device(api_device)
        np.testing.assert_array_equal(parameter.numpy(), again.numpy())

    assert not np.array_equal(first.parameters()[0].numpy(), changed.parameters()[0].numpy())

def test_physics_loss_does_not_depend_on_network_structure(pinn_example, api_device):
    points = om.Tensor(np.linspace(0.0, np.pi / 2, 8).reshape(-1, 1), om.float64, api_device)
    loss = pinn_example["pinn_loss"](nn.Sin(), points)
    assert not loss.realized
    assert loss.item() == pytest.approx(0.0, abs=1e-24)

def test_example_training_uses_the_module_parameters(pinn_example, api_device, capsys):
    arguments = argparse.Namespace(
        epochs=5, points=4, width=3, learning_rate=0.01,
        device=api_device, report_every=5, seed=7,
        no_progress=True,
    )
    model = pinn_example["train"](arguments)
    losses = [float(line.split()[-1]) for line in capsys.readouterr().out.splitlines()]
    assert isinstance(model, nn.Module)
    assert len(losses) == 2
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    assert all(parameter.realized and parameter.op == "leaf" for parameter in model.parameters())
