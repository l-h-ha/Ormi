import gc
import weakref

import numpy as np
import pytest

import ormi_core as oc

###
###
###

# Positive log/rsqrt inputs and bounded exponentials keep numerical checks stable.
OPERATIONS = ("sin", "cos", "exp", "log", "rsqrt", "mul", "div", "center")

def _finite_gradient(function, values, step=1e-5):
    """Estimate gradients from scalar NumPy evaluations."""

    gradients = []

    for number, value in enumerate(values):
        gradient = np.empty_like(value)

        for index in np.ndindex(value.shape):
            upper = [array.copy() for array in values]
            lower = [array.copy() for array in values]
            upper[number][index] += step
            lower[number][index] -= step
            gradient[index] = (function(upper) - function(lower)) / (2 * step)

        gradients.append(gradient)

    return gradients

def _finite_hessian_product(function, values, directions, step=2e-4):
    """Estimate mixed second derivatives along an input direction."""

    upper = [value + step * direction for value, direction in zip(values, directions, strict=True)]
    lower = [value - step * direction for value, direction in zip(values, directions, strict=True)]
    upper_gradient = _finite_gradient(function, upper, step)
    lower_gradient = _finite_gradient(function, lower, step)
    return [(high - low) / (2 * step)
            for high, low in zip(upper_gradient, lower_gradient, strict=True)]

def _check_derivatives(build, reference, batches, device, rng):
    inputs = [oc.Tensor(value, oc.float64, device) for value in batches[0]]
    weights = rng.normal(size=np.shape(reference(batches[0])))
    seed = oc.Tensor(weights, oc.float64, device)
    directions = [rng.normal(size=value.shape) for value in batches[0]]
    length = np.sqrt(sum(np.sum(direction * direction) for direction in directions))
    directions = [direction / length for direction in directions]
    output = build(inputs)
    loss = (output * seed).sum()
    gradients = oc.grad(output, inputs, seed)
    directional = (gradients[0] * oc.Tensor(directions[0], oc.float64, device)).sum()

    for gradient, direction in zip(gradients[1:], directions[1:], strict=True):
        directional = directional + (gradient * oc.Tensor(direction, oc.float64, device)).sum()

    hessian_product = oc.grad(directional, inputs)
    roots = [output, loss, *gradients, *hessian_product]
    assert all(not root.realized for root in roots)
    plan = oc.Plan(roots)
    assert all(not root.realized for root in roots)
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    def objective(values):
        return float(np.sum(reference(values) * weights))

    def check_values(values, batch, stage):
        context = f"batch={batch}, stage={stage}"

        for number, root in enumerate(roots):
            assert np.isfinite(root.numpy()).all(), f"Non-finite value, {context}, root={number}"

        np.testing.assert_allclose(
            output.numpy(), reference(values), rtol=1e-11, atol=1e-12, equal_nan=False, err_msg=context,
        )
        np.testing.assert_allclose(
            loss.item(), objective(values), rtol=1e-11, atol=1e-12, equal_nan=False, err_msg=context,
        )
        expected_gradient = _finite_gradient(objective, values)
        expected_hessian = _finite_hessian_product(objective, values, directions)

        for number, (actual, expected) in enumerate(zip(gradients, expected_gradient, strict=True)):
            np.testing.assert_allclose(
                actual.numpy(), expected, rtol=2e-6, atol=2e-8, equal_nan=False,
                err_msg=f"gradient, batch={batch}, input={number}, stage={stage}",
            )

        for number, (actual, expected) in enumerate(zip(hessian_product, expected_hessian, strict=True)):
            np.testing.assert_allclose(
                actual.numpy(), expected, rtol=3e-4, atol=2e-6, equal_nan=False,
                err_msg=f"Hessian product, batch={batch}, input={number}, stage={stage}",
            )

    for batch, values in enumerate(batches):
        for tensor, value in zip(inputs, values, strict=True):
            tensor.copy_from(oc.Tensor(value, oc.float64, device))

        plan.run()
        check_values(values, batch, "plan")
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    if device == "cpu":
        return

    before_recording = [root.numpy().copy() for root in roots]
    replay = plan.record()

    for root, expected in zip(roots, before_recording, strict=True):
        np.testing.assert_array_equal(root.numpy(), expected, err_msg="recording changed an output")

    for batch, values in enumerate(batches):
        for tensor, value in zip(inputs, values, strict=True):
            tensor.copy_from(oc.Tensor(value, oc.float64, device))

        replay.run()
        check_values(values, batch, "replay")
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

    plan_reference = weakref.ref(plan)
    del plan
    gc.collect()
    assert plan_reference() is None

    for batch, values in enumerate(batches):
        for tensor, value in zip(inputs, values, strict=True):
            tensor.copy_from(oc.Tensor(value, oc.float64, device))

        replay.run()
        check_values(values, batch, "replay without Plan")

def _ormi_expression(inputs, operations):
    x, scale, bias = inputs
    view = x.transpose((2, 0, 1))
    expanded = scale.expand(view.shape)
    assert not view.is_contiguous() and not expanded.is_contiguous()
    shared = view * expanded + bias
    shared = shared + shared.mean((0,), keepdims=True)
    value = shared * shared + shared.cos()

    for operation in operations:
        if operation == "sin":
            value = value.sin()
        elif operation == "cos":
            value = value.cos()
        elif operation == "exp":
            value = (value * 0.2).exp()
        elif operation == "log":
            value = (value * value + 1.25).log()
        elif operation == "rsqrt":
            value = (value * value + 1.25).rsqrt()
        elif operation == "mul":
            value = value * shared
        elif operation == "div":
            value = value / (shared * shared + 1.25)
        elif operation == "center":
            value = value - value.mean((0, 2), keepdims=True)
        else:
            raise AssertionError(operation)

    return value + shared.sin() * value.mean()

def _numpy_expression(values, operations):
    x, scale, bias = values
    view = x.transpose(2, 0, 1)
    shared = view * np.broadcast_to(scale, view.shape) + bias
    shared = shared + shared.mean(axis=0, keepdims=True)
    value = shared**2 + np.cos(shared)

    for operation in operations:
        if operation == "sin":
            value = np.sin(value)
        elif operation == "cos":
            value = np.cos(value)
        elif operation == "exp":
            value = np.exp(value * 0.2)
        elif operation == "log":
            value = np.log(value**2 + 1.25)
        elif operation == "rsqrt":
            value = 1 / np.sqrt(value**2 + 1.25)
        elif operation == "mul":
            value = value * shared
        elif operation == "div":
            value = value / (shared**2 + 1.25)
        elif operation == "center":
            value = value - value.mean(axis=(0, 2), keepdims=True)
        else:
            raise AssertionError(operation)

    return value + np.sin(shared) * value.mean()

def test_finite_difference_reference_includes_cross_input_derivatives():
    rng = np.random.default_rng(31337)
    matrix = rng.normal(size=(5, 5))
    matrix = matrix + matrix.T
    values = [rng.normal(size=2), rng.normal(size=3)]
    directions = [rng.normal(size=2), rng.normal(size=3)]

    def quadratic(arrays):
        vector = np.concatenate(arrays)
        return float(0.5 * vector @ matrix @ vector)

    expected_gradient = matrix @ np.concatenate(values)
    expected_hessian = matrix @ np.concatenate(directions)
    np.testing.assert_allclose(
        np.concatenate(_finite_gradient(quadratic, values)), expected_gradient,
        rtol=1e-9, atol=2e-10,
    )
    np.testing.assert_allclose(
        np.concatenate(_finite_hessian_product(quadratic, values, directions)), expected_hessian,
        rtol=1e-7, atol=2e-7,
    )

@pytest.mark.parametrize("wrong_order,message", [(1, "gradient, batch=0"), (2, "Hessian product, batch=0")])
def test_checker_rejects_incorrect_derivatives(monkeypatch, wrong_order, message):
    rng = np.random.default_rng(43)
    values = [[rng.uniform(-0.5, 0.5, size=(2, 3))]]
    original_grad = oc.grad
    calls = 0

    def incorrect_grad(output, inputs, seed=None):
        nonlocal calls
        gradients = original_grad(output, inputs, seed)
        calls += 1

        if calls == wrong_order:
            return [-gradient for gradient in gradients]

        return gradients

    def build(inputs):
        return inputs[0] * inputs[0]

    def reference(arrays):
        return arrays[0]**2

    monkeypatch.setattr(oc, "grad", incorrect_grad)

    with pytest.raises(AssertionError, match=message):
        _check_derivatives(build, reference, values, "cpu", rng)

@pytest.mark.parametrize("seed", range(len(OPERATIONS)))
def test_randomized_compositions_match_numerical_derivatives(nn_device, seed):
    rng = np.random.default_rng(31337 + seed)
    batch, channels, features = rng.integers(2, 4, size=3)
    shapes = [(channels, features, batch), (channels, 1), (features,)]
    batches = [[rng.uniform(-0.5, 0.5, size=shape) for shape in shapes] for _ in range(2)]
    operations = [OPERATIONS[seed], *rng.choice(OPERATIONS, size=3)]

    def build(inputs):
        return _ormi_expression(inputs, operations)

    def reference(values):
        return _numpy_expression(values, operations)

    _check_derivatives(build, reference, batches, nn_device, rng)

@pytest.mark.parametrize("case,shapes", [
    ("dot", [(3,), (3,)]),
    ("matrix_vector", [(2, 3, 4), (4,)]),
    ("vector_matrix", [(3,), (2, 3, 4)]),
    ("broadcast_batches", [(2, 1, 2, 3), (1, 3, 3, 2)]),
    ("transposed", [(2, 3), (2, 2)]),
])
@pytest.mark.parametrize("seed", [17, 29])
def test_randomized_matmul_matches_numerical_derivatives(nn_device, case, shapes, seed):
    rng = np.random.default_rng(seed)
    batches = [[rng.uniform(-0.5, 0.5, size=shape) for shape in shapes] for _ in range(2)]

    def build(inputs):
        lhs, rhs = inputs

        if case == "transposed":
            lhs = lhs.transpose((1, 0))

        product = lhs @ rhs
        shared = product.sin() + product * product
        return shared.cos() / (product * product + 1.5)

    def reference(values):
        lhs, rhs = values

        if case == "transposed":
            lhs = lhs.T

        product = lhs @ rhs
        shared = np.sin(product) + product**2
        return np.cos(shared) / (product**2 + 1.5)

    _check_derivatives(build, reference, batches, nn_device, rng)

def test_randomized_shared_graph_preserves_disconnected_gradients(nn_device):
    rng = np.random.default_rng(571)
    batches = [[rng.uniform(-0.5, 0.5, size=shape) for shape in [(2, 3), (2,)]] for _ in range(2)]

    def build(inputs):
        x, unused = inputs
        shared = x.sin() * x
        return shared * shared + shared

    def reference(values):
        x, unused = values
        shared = np.sin(x) * x
        return shared**2 + shared

    _check_derivatives(build, reference, batches, nn_device, rng)
