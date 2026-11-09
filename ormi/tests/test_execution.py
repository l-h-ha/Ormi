import numpy as np
import pytest

import ormi as om
import ormi_core as core
from ormi import nn

###
###
###

def test_mixed_imports_preserve_lazy_higher_order_derivatives(api_device):
    x = core.Tensor(2.0, core.float64, api_device)
    y = om.ops.mul(x * x, x * x)
    first = om.diff(y, x)
    second = core.diff(first, x)
    third = om.diff(second, x)
    roots = [y, first, second, third]

    assert type(x) is om.Tensor
    assert all(type(value) is core.Tensor and not value.realized for value in roots)
    plan = om.Plan(roots)
    assert all(not value.realized for value in roots)
    plan.run()
    assert [value.item() for value in roots] == [16.0, 32.0, 48.0, 48.0]

def test_linear_preserves_all_leading_axes(api_device):
    values = np.arange(24, dtype=np.float64).reshape(2, 1, 4, 3) / 10
    weight = core.Tensor([[1.0, -1.0], [2.0, 0.5], [3.0, 2.0]], core.float64, api_device)
    bias = om.Tensor([0.25, -0.5], om.float64, api_device)
    model = nn.Sequential([nn.Linear(weight, bias)])
    output = model(om.Tensor(values, om.float64, api_device))

    assert isinstance(model, core.nn.Module)
    assert model.parameters()[0] is weight
    assert model.parameters()[1] is bias
    assert output.shape == (2, 1, 4, 2)
    assert not output.realized
    np.testing.assert_allclose(output.numpy(), values @ weight.numpy() + bias.numpy(), atol=1e-12)

def test_training_reuses_the_core_plan_and_parameter_objects(api_device):
    weight = core.Tensor([[2.0]], core.float64, api_device)
    layer = nn.Linear(weight)
    input = om.ones((2, 1), om.float64, api_device)
    loss = nn.mse_loss(layer(input), core.zeros_like(input))
    gradient = om.grad(loss, [weight])[0]
    update = om.sgd(weight, gradient, 0.1)
    assert not loss.realized and not gradient.realized and not update.realized
    plan = om.Plan([loss], [(weight, update)])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    for step in range(2):
        plan.run()
        assert loss.item() == pytest.approx((2 * 0.8 ** step) ** 2)
        assert weight.item() == pytest.approx(2 * 0.8 ** (step + 1))
        assert layer.parameters()[0] is weight
        assert weight.realized and weight.op == "leaf"
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    if api_device == "cpu":
        return

    replay = plan.record()
    assert type(replay) is core.Replay
    assert weight.item() == pytest.approx(1.28)
    del plan, layer

    for step in range(2, 5):
        replay.run()
        assert loss.item() == pytest.approx((2 * 0.8 ** step) ** 2)
        assert weight.item() == pytest.approx(2 * 0.8 ** (step + 1))
