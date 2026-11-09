from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import ormi_core as oc
from ormi_core import nn

###
###
###

class Identity(nn.Module):
    def forward(self, input):
        return input

###
###
###

def test_module_training_modes_and_buffers_are_recursive_and_separate():
    child = Identity()
    buffer = oc.Tensor([1.0])
    parameter = oc.Tensor([2.0])
    child.register_buffer(buffer)
    child.register_buffer(buffer)
    child.register_parameter(parameter)
    nested = nn.Sequential([child])
    root = nn.Sequential([nested, child])
    assert root.buffers() == [buffer] and root.parameters() == [parameter]
    assert all(layer.training for layer in [root, nested, child])
    root.eval()
    assert not any(layer.training for layer in [root, nested, child])
    root.train()
    assert all(layer.training for layer in [root, nested, child])
    root.train(False)
    assert not child.training

    with pytest.raises(oc.OperationError):
        child.register_buffer(parameter + 1)

class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.register_module(layer)

    def forward(self, input):
        return self.layer(input).sin() + input

def test_custom_module_preserves_batch_axes_and_fifth_order_gradients(nn_device):
    values = np.arange(36, dtype=np.float64).reshape(2, 1, 3, 2, 1, 3) / 10
    weight = oc.Tensor(np.arange(9).reshape(3, 3) / 10, oc.float64, nn_device)
    bias = oc.Tensor([-0.1, 0.2, 0.3], oc.float64, nn_device)
    input = oc.Tensor(values, oc.float64, nn_device)
    model = Residual(nn.Linear(weight, bias))
    sequence = nn.Sequential([model])
    output = sequence(input)
    manual = (input @ weight + bias).sin() + input
    derivatives = [output]
    manual_derivatives = [manual]

    for _ in range(5):
        derivatives.append(oc.grad(derivatives[-1].sum(), [input])[0])
        manual_derivatives.append(oc.grad(manual_derivatives[-1].sum(), [input])[0])

    loss = nn.mse_loss(derivatives[2], oc.zeros_like(derivatives[2]))
    manual_loss = nn.mse_loss(manual_derivatives[2], oc.zeros_like(manual_derivatives[2]))
    parameters = sequence.parameters()
    assert parameters == [weight, bias]
    roots = [*derivatives, loss, *oc.grad(loss, parameters)]
    manual_roots = [*manual_derivatives, manual_loss, *oc.grad(manual_loss, parameters)]
    assert all(not tensor.realized for tensor in roots)
    assert all(tensor.shape == values.shape for tensor in derivatives)

    plan = oc.Plan(roots)
    manual_plan = oc.Plan(manual_roots)
    assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == (
        manual_plan.steps, manual_plan.storage_buffers, manual_plan.storage_bytes
    )
    plan.run()
    manual_plan.run()

    for actual, expected in zip(roots, manual_roots, strict=True):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-11, atol=1e-12)

    np.testing.assert_allclose(output.numpy(), np.sin(values @ weight.numpy() + bias.numpy()) + values)

def test_prepared_training_does_not_call_python_forward_again(nn_device):
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.weight = oc.Tensor([2.0], oc.float64, nn_device)
            self.register_parameter(self.weight)

        def forward(self, input):
            self.calls += 1
            return input * self.weight

    layer = Scale()
    model = nn.Sequential([layer])
    input = oc.Tensor(np.ones((2, 3, 1)), oc.float64, nn_device)
    loss = nn.mse_loss(model(input), oc.zeros_like(input))
    weight = model.parameters()[0]
    gradient = oc.grad(loss, [weight])[0]
    plan = oc.Plan([loss], [(weight, oc.sgd(weight, gradient, 0.1))])
    storage = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    for step in range(8):
        plan.run()
        np.testing.assert_allclose(weight.numpy(), [2.0 * 0.8 ** (step + 1)], rtol=1e-12)
        assert layer.calls == 1
        assert layer.parameters()[0] is weight
        assert weight.realized and weight.op == "leaf"
        assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == storage

def test_cpp_sequences_retain_python_layers_and_release_them_after_use(nn_device):
    layer = Residual(nn.Linear(3, 3, dtype=oc.float64, device=nn_device, seed=7))
    lifetime = weakref.ref(layer)
    sequence = nn.Sequential()
    sequence.append(layer)
    model = nn.Sequential([sequence])
    input = oc.Tensor(np.ones((2, 1, 3)), oc.float64, nn_device)
    expected = (input @ layer.layer.weight + layer.layer.bias).sin() + input

    del layer, sequence
    gc.collect()
    assert lifetime() is not None
    assert model.children()[0].children()[0] is lifetime()
    output = model(input)

    del model
    gc.collect()
    assert lifetime() is None
    np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)

def test_custom_module_registration_is_explicit_and_deduplicates_shared_objects(nn_device):
    first = Identity()
    second = Identity()
    weight = oc.Tensor([2.0], oc.float64, nn_device)
    first.weight = weight
    first.other = second
    assert first.parameters() == []
    assert first.children() == []

    first.register_parameter(weight)
    second.register_parameter(weight)
    first.register_module(second)
    first.register_module(second)
    assert first.parameters() == [weight]
    assert first.children() == [second, second]

def test_custom_modules_reject_ownership_cycles_without_retaining_them():
    first = Identity()
    second = Identity()
    first_lifetime = weakref.ref(first)
    second_lifetime = weakref.ref(second)
    first.register_module(second)

    with pytest.raises(oc.OperationError, match="cycle"):
        second.register_module(first)

    with pytest.raises(oc.OperationError, match="cycle"):
        first.register_module(first)

    with pytest.raises(oc.OperationError, match="null"):
        first.register_module(None)

    assert second.children() == []
    assert first.children() == [second]

    del first, second
    gc.collect()
    assert first_lifetime() is None
    assert second_lifetime() is None

def test_custom_module_rejects_invalid_parameters(nn_device):
    layer = Identity()
    input = oc.Tensor([1.0], oc.float64, nn_device)
    computed = input.sin()

    with pytest.raises(oc.OperationError, match="null"):
        layer.register_parameter(None)

    with pytest.raises(oc.OperationError, match="leaf"):
        layer.register_parameter(computed)

    computed.numpy()

    with pytest.raises(oc.OperationError, match="leaf"):
        layer.register_parameter(computed)

    with pytest.raises(oc.DTypeError, match="floating-point"):
        layer.register_parameter(oc.Tensor([1], oc.int32, nn_device))

    assert layer.parameters() == []

def test_python_forward_exceptions_leave_sequence_usable(nn_device):
    class BrokenLayer(ValueError):
        pass

    class Recoverable(nn.Module):
        def __init__(self):
            super().__init__()
            self.fail = True

        def forward(self, input):
            if self.fail:
                raise BrokenLayer("forward failed")

            return input

    layer = Recoverable()
    model = nn.Sequential([layer])
    input = oc.Tensor([1.0], device=nn_device)

    with pytest.raises(BrokenLayer, match="forward failed"):
        model(input)

    layer.fail = False
    assert model(input) is input

    with pytest.raises(oc.OperationError, match="input tensor"):
        model(None)

def test_appending_during_forward_takes_effect_on_the_next_call(nn_device):
    model = nn.Sequential()
    owner = weakref.ref(model)

    class AppendOnce(nn.Module):
        def __init__(self):
            super().__init__()
            self.appended = False

        def forward(self, input):
            if not self.appended:
                for _ in range(32):
                    owner().append(nn.Sin())

                self.appended = True

            return input

    model.append(AppendOnce())
    values = np.array([0.25, 0.5], dtype=np.float64)
    input = oc.Tensor(values, oc.float64, nn_device)
    assert model(input) is input
    assert len(model.children()) == 33

    output = model(input)
    assert not output.realized

    for _ in range(32):
        values = np.sin(values)

    np.testing.assert_allclose(output.numpy(), values, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("result", [None, 42, [1.0], np.array([1.0])])
def test_custom_module_must_return_a_tensor(result):
    class Invalid(nn.Module):
        def forward(self, input):
            return result

    with pytest.raises(oc.OperationError, match="must return a Tensor"):
        nn.Sequential([Invalid()])(oc.Tensor([1.0]))

def test_python_module_must_implement_forward():
    input = oc.Tensor([1.0])

    with pytest.raises(oc.OperationError, match="implement forward"):
        nn.Module()(input)

    with pytest.raises(oc.OperationError, match="implement forward"):
        nn.Sequential([nn.Module()])(input)

def test_python_module_must_initialize_its_cpp_base():
    class Invalid(nn.Module):
        def __init__(self):
            pass

        def forward(self, input):
            return input

    with pytest.raises(TypeError, match="__init__"):
        Invalid()

@pytest.mark.parametrize("layer", [nn.Linear, nn.Sin, nn.Sequential])
def test_builtin_layers_remain_final(layer):
    with pytest.raises(TypeError):
        type("Invalid", (layer,), {})
