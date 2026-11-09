from __future__ import annotations

import gc
import os
import subprocess
import sys

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

def test_nn_exports_and_cpp_inheritance():
    from ormi_core.nn import Linear, Module, Sequential, Sin, mse_loss

    assert Linear is nn.Linear
    assert issubclass(Linear, Module)
    assert issubclass(Sin, Module)
    assert issubclass(Sequential, Module)
    assert callable(mse_loss)

@pytest.mark.parametrize(
    "shape", [(3,), (4, 3), (2, 4, 3), (1, 2, 4, 3), (2, 1, 3, 2, 1, 3), (0, 3)]
)
def test_linear_matches_numpy_for_all_batch_ranks(nn_device, shape):
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10
    weights = np.array([[1, -1], [2, 0.5], [3, 2]], dtype=np.float64)
    biases = np.array([0.25, -0.5], dtype=np.float64)
    weight = oc.Tensor(weights, oc.float64, nn_device)
    bias = oc.Tensor(biases, oc.float64, nn_device)
    layer = nn.Linear(weight, bias)
    output = layer(oc.Tensor(values, oc.float64, nn_device))

    assert not output.realized
    assert layer.weight is weight
    assert layer.bias is bias
    assert layer.in_features == 3
    assert layer.out_features == 2
    assert layer.parameters() == [weight, bias]
    np.testing.assert_allclose(output.numpy(), values @ weights + biases, rtol=1e-12, atol=1e-12)

def test_linear_accepts_strided_inputs_and_optional_bias(nn_device):
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    layer = nn.Linear(oc.Tensor([[1], [2], [3]], oc.float64, nn_device))
    input = oc.Tensor(values, oc.float64, nn_device).transpose((1, 0))
    output = layer.forward(input)
    assert layer.bias is None
    assert len(layer.parameters()) == 1
    assert not output.realized
    np.testing.assert_allclose(output.numpy(), values.T @ np.array([[1, 2, 3]]).T)

def test_linear_preserves_batch_axes_through_higher_order_gradients(nn_device):
    values = np.arange(36, dtype=np.float64).reshape(2, 1, 3, 2, 1, 3) / 10
    weights = np.array([[1, -1], [2, 0.5], [3, 2]], dtype=np.float64)
    biases = np.array([0.25, -0.5], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    weight = oc.Tensor(weights, oc.float64, nn_device)
    bias = oc.Tensor(biases, oc.float64, nn_device)
    output = nn.Linear(weight, bias)(input)
    loss = (output * output).sum()
    input_gradient, weight_gradient, bias_gradient = oc.grad(loss, [input, weight, bias])
    second = oc.grad(input_gradient.sum(), [input])[0]
    roots = [output, input_gradient, weight_gradient, bias_gradient, second]
    assert all(not tensor.realized for tensor in roots)
    oc.Plan(roots).run()

    expected = values @ weights + biases
    expected_weight_gradient = values.reshape(-1, 3).T @ (2 * expected.reshape(-1, 2))
    batch_axes = tuple(range(expected.ndim - 1))
    expected_second = np.broadcast_to(2 * weights @ weights.sum(axis=0), values.shape)
    np.testing.assert_allclose(output.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(input_gradient.numpy(), 2 * expected @ weights.T, atol=1e-12)
    np.testing.assert_allclose(weight_gradient.numpy(), expected_weight_gradient, atol=1e-12)
    np.testing.assert_allclose(bias_gradient.numpy(), (2 * expected).sum(axis=batch_axes), atol=1e-12)
    np.testing.assert_allclose(second.numpy(), expected_second, atol=1e-12)

@pytest.mark.parametrize("dtype", [oc.float16, oc.float32, oc.float64])
def test_linear_initialization_is_seeded_and_creates_leaf_parameters(nn_device, dtype):
    first = nn.Linear(3, 5, dtype=dtype, device=nn_device, seed=17)
    repeated = nn.Linear(3, 5, dtype=dtype, device=oc.Device(nn_device), seed=17)
    other = nn.Linear(3, 5, dtype=dtype, device=nn_device, seed=18)
    assert first.weight.shape == (3, 5)
    assert first.bias.shape == (5,)

    for actual, again in zip(first.parameters(), repeated.parameters(), strict=True):
        assert actual.realized
        assert actual.op == "leaf"
        assert actual.dtype == dtype
        assert actual.device == oc.Device(nn_device)
        np.testing.assert_array_equal(actual.numpy(), again.numpy())
        assert np.max(np.abs(actual.numpy())) <= 1 / np.sqrt(3) + 1e-3

    assert not np.array_equal(first.weight.numpy(), other.weight.numpy())
    assert nn.Linear(3, 5, bias=False, device=nn_device, seed=17).bias is None

def test_sequential_owns_children_and_deduplicates_shared_parameters(nn_device):
    first = nn.Linear(2, 2, dtype=oc.float64, device=nn_device, seed=1)
    second = nn.Linear(first.weight, first.bias)
    model = nn.Sequential([first, nn.Sequential([nn.Sin(), second]), first])
    parameters = model.parameters()
    assert parameters == [first.weight, first.bias]
    assert len(model.children()) == 3
    del first, second
    gc.collect()
    output = model(oc.Tensor([[0.25, -0.5]], oc.float64, nn_device))
    assert not output.realized
    assert np.isfinite(output.numpy()).all()

def test_sequences_reject_cycles_and_preserve_empty_identity(nn_device):
    first = nn.Sequential()
    input = oc.Tensor([1.0], device=nn_device)
    assert first(input) is input
    second = nn.Sequential([first])

    with pytest.raises(oc.OperationError, match="cycle"):
        first.append(second)

    with pytest.raises(oc.OperationError, match="cycle"):
        first.append(first)

    with pytest.raises(oc.OperationError, match="null"):
        first.append(None)

    assert first.children() == []

def test_network_matches_manual_graph_and_fifth_order_input_derivatives(nn_device):
    values = np.array([[-0.3], [0.0], [0.4]], dtype=np.float64)
    weights = np.array([[0.3, -0.7, 1.2]], dtype=np.float64)
    biases = np.array([-0.1, 0.2, 0.4], dtype=np.float64)
    output_weights = np.array([[0.5], [-0.2], [0.3]], dtype=np.float64)
    first = nn.Linear(
        oc.Tensor(weights, oc.float64, nn_device),
        oc.Tensor(biases, oc.float64, nn_device),
    )
    second = nn.Linear(
        oc.Tensor(output_weights, oc.float64, nn_device),
        oc.Tensor([0.05], oc.float64, nn_device),
    )
    model = nn.Sequential([first, nn.Sin(), second])
    input = oc.Tensor(values, oc.float64, nn_device)
    output = model(input)
    manual = (input @ first.weight + first.bias).sin()
    manual = manual @ second.weight + second.bias
    derivatives = [output]
    manual_derivatives = [manual]

    for _ in range(5):
        derivatives.append(oc.grad(derivatives[-1].sum(), [input])[0])
        manual_derivatives.append(oc.grad(manual_derivatives[-1].sum(), [input])[0])

    residual = derivatives[2] + output
    manual_residual = manual_derivatives[2] + manual
    loss = nn.mse_loss(residual, oc.zeros_like(residual))
    manual_error = manual_residual - oc.zeros_like(manual_residual)
    manual_loss = (manual_error * manual_error).mean()
    gradients = oc.grad(loss, model.parameters())
    manual_gradients = oc.grad(manual_loss, model.parameters())
    roots = [*derivatives, loss, *gradients]
    manual_roots = [*manual_derivatives, manual_loss, *manual_gradients]
    assert all(not value.realized for value in roots)
    plan = oc.Plan(roots)
    manual_plan = oc.Plan(manual_roots)
    assert plan.steps == manual_plan.steps
    assert plan.storage_buffers == manual_plan.storage_buffers
    assert plan.storage_bytes == manual_plan.storage_bytes
    plan.run()
    manual_plan.run()

    for actual, expected in zip(roots, manual_roots, strict=True):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-11, atol=1e-12)

    phase = values @ weights + biases
    np.testing.assert_allclose(output.numpy(), np.sin(phase) @ output_weights + 0.05, atol=1e-12)

    for order in range(1, 6):
        expected = (np.sin(phase + order * np.pi / 2) * weights**order) @ output_weights
        np.testing.assert_allclose(derivatives[order].numpy(), expected, rtol=1e-11, atol=1e-12)

def test_prepared_training_keeps_parameter_objects_and_storage(nn_device):
    weight = oc.Tensor([[2.0]], oc.float64, nn_device)
    layer = nn.Linear(weight)
    input = oc.Tensor(np.ones((4, 1)), oc.float64, nn_device)
    target = oc.Tensor(np.zeros((4, 1)), oc.float64, nn_device)
    loss = nn.mse_loss(layer(input), target)
    gradient = oc.grad(loss, [weight])[0]
    plan = oc.Plan([loss], [(weight, oc.sgd(weight, gradient, 0.1))])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    for step in range(8):
        plan.run()
        np.testing.assert_allclose(weight.numpy(), [[2 * 0.8 ** (step + 1)]], rtol=1e-12)
        assert layer.parameters()[0] is weight
        assert weight.realized
        assert weight.op == "leaf"
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

def test_output_graph_survives_model_destruction(nn_device):
    layer = nn.Linear(
        oc.Tensor([[3.0], [3.0]], oc.float64, nn_device),
        oc.Tensor([0.5], oc.float64, nn_device),
    )
    output = layer(oc.Tensor([2.0, 2.0], oc.float64, nn_device))
    del layer
    gc.collect()
    np.testing.assert_array_equal(output.numpy(), [12.5])

def test_autocast_uses_the_same_tensor_operations(nn_device):
    layer = nn.Linear(3, 2, device=nn_device, seed=7)
    input = oc.Tensor([[0.1, -0.2, 0.3]], device=nn_device)

    with oc.autocast():
        actual = layer(input)
        manual = input @ layer.weight + layer.bias

    assert actual.dtype == manual.dtype
    np.testing.assert_array_equal(actual.numpy(), manual.numpy())

def test_mse_loss_is_lazy_and_twice_differentiable(nn_device):
    prediction = oc.Tensor([3.0, 3.0], oc.float64, nn_device)
    target = oc.Tensor([1.0, 1.0], oc.float64, nn_device)
    loss = nn.mse_loss(prediction, target)
    first = oc.grad(loss, [prediction])[0]
    second = oc.grad(first.sum(), [prediction])[0]
    assert not loss.realized and not first.realized and not second.realized
    assert loss.item() == 4.0
    np.testing.assert_array_equal(first.numpy(), [2.0, 2.0])
    np.testing.assert_array_equal(second.numpy(), [1.0, 1.0])

    with pytest.raises(oc.ShapeError):
        nn.mse_loss(prediction, oc.Tensor([1.0], oc.float64, nn_device))

    with pytest.raises(oc.DTypeError):
        nn.mse_loss(prediction, oc.Tensor([1, 1], oc.int32, nn_device))

def test_invalid_linear_arguments_fail_before_execution():
    with pytest.raises(oc.ShapeError):
        nn.Linear(0, 2)

    with pytest.raises(oc.ShapeError):
        nn.Linear(2, -1)

    with pytest.raises(oc.DTypeError):
        nn.Linear(2, 3, dtype=oc.int32)

    with pytest.raises(oc.ShapeError):
        nn.Linear(oc.Tensor([1.0, 2.0]))

    with pytest.raises(oc.OperationError, match="leaf"):
        nn.Linear(oc.Tensor([[1.0]]).sin())

    layer = nn.Linear(2, 3, seed=7)

    with pytest.raises(oc.ShapeError):
        layer(oc.Tensor(1.0))

    with pytest.raises(oc.ShapeError):
        layer(oc.Tensor([1.0]))

    with pytest.raises(oc.OperationError):
        layer(None)

def test_initialization_and_graph_construction_do_not_resolve_a_compiler():
    environment = os.environ.copy()
    environment["ORMI_CPU_JIT_CXX"] = "ormi-compiler-does-not-exist"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ormi_core as oc; from ormi_core import nn; "
            "m = nn.Sequential([nn.Linear(2, 3, seed=1), nn.Sin()]); "
            "y = m(oc.Tensor([1., 2.])); assert not y.realized",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
