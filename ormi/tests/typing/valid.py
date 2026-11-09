from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import assert_type

import ormi as om
import ormi.nn as nn
import ormi_core as core
from ormi import jit, ops, optim
from ormi.jit import CacheInfo
from ormi.nn import Linear, Module
from ormi.ops import matmul
from ormi.optim import GradScaler

###
###
###

class Residual(Module):
    def __init__(self, width: int):
        super().__init__()
        self.layer = Linear(width, width, seed=7)
        self.register_module(self.layer, name="layer")

    def forward(self, input: om.Tensor) -> om.Tensor:
        return self.layer(input).sin() + input

x = om.tensor([[[1.0, 2.0, 3.0]]], dtype=om.float64)
weight = core.Tensor([[1.0], [2.0], [3.0]], dtype=core.float64)
layer = Linear(weight)
network = nn.Sequential([Residual(3), nn.Linear(3, 1, seed=8)])
output = network(x)
loss = nn.mse_loss(output, om.zeros_like(output))
gradient = om.grad(loss, [x])[0]
single_gradient = om.grad(loss, x)
point_derivative = om.partial(output, x, coordinate=1, order=2)
second = om.diff(gradient.sum(), x)
update = optim.sgd(weight, core.grad(layer(x).sum(), [weight])[0], 0.1)
plan = om.Plan([loss, gradient, second], [(weight, update)])

assert_type(x, core.Tensor)
assert_type(x.copy_from(core.Tensor([[[1.0, 2.0, 3.0]]], dtype=core.float64)), None)
assert_type(output, om.Tensor)
assert_type(layer, core.nn.Linear)
assert_type(network.parameters(), list[core.Tensor])
assert_type(network.children(), list[core.nn.Module])
assert_type(network.state(), dict[str, core.Tensor])
assert_type(gradient, core.Tensor)
assert_type(single_gradient, core.Tensor)
assert_type(loss.grad(x), core.Tensor)
assert_type(loss.grad([x]), list[core.Tensor])
assert_type(om.grad(loss, [x]), list[core.Tensor])
assert_type(point_derivative, core.Tensor)
assert_type(second, core.Tensor)
assert_type(x.shape, tuple[int, ...])
assert_type(x.device, core.Device)
assert_type(x.dtype, core.DType)
assert_type(ops.matmul(x, weight), core.Tensor)
assert_type(matmul(x, weight), om.Tensor)
assert_type(plan, core.Plan)
assert_type(plan.record(), core.Replay)
assert_type(jit.info(), CacheInfo)
assert_type(GradScaler(1024.0).scale(loss), om.Tensor)

with om.autocast():
    assert_type(layer(x), core.Tensor)

def checkpoint_types(model: Module, path: str) -> None:
    assert_type(om.save(model, path), None)
    assert_type(om.load(model, path), None)

dataset = np.zeros((7, 2, 3), dtype=np.float32)
assert_type(om.batches(dataset, batch_size=3, shuffle=True, seed=7), Iterator[tuple[NDArray[Any], ...]])
assert_type(
    om.batches(dataset, batch_size=3, shuffle=True, rng=np.random.default_rng(7), drop_last=True),
    Iterator[tuple[NDArray[Any], ...]],
)

def training_types(model: Module, values: NDArray[Any]) -> None:
    def objective(inputs: om.Tensor) -> om.Tensor:
        return model(inputs).mean()

    assert_type(
        om.fit(model, values, loss=objective,
               update=lambda p, g: om.sgd(p, g, 0.01), epochs=3, verbose=0),
        dict[int, float],
    )
