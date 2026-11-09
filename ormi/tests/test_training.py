from copy import deepcopy
import io

import numpy as np
import pytest
from tqdm import auto as tqdm_auto
from tqdm.auto import tqdm

import ormi as om
import ormi.training as training
from ormi import nn

###
###
###

def test_fit_reuses_plans_and_matches_numpy_sgd_with_arbitrary_rank(api_device, monkeypatch):
    values = np.sin(np.arange(42)).reshape(7, 1, 3, 2)
    targets = np.cos(np.arange(21)).reshape(7, 1, 3, 1)
    weights = np.array([[0.2], [-0.1]])
    expected = weights.copy()
    weight = om.tensor(weights, om.float64, api_device)
    model = nn.Linear(weight)
    model.eval()
    prepared = []
    plans = []
    updates = []
    plan_type = training.Plan

    def loss(x, y):
        result = nn.mse_loss(model(x), y)
        assert not result.realized
        prepared.append((x, y))
        return result

    def update(parameter, gradient):
        assert parameter is weight and not gradient.realized
        updates.append(gradient)
        return om.sgd(parameter, gradient, 0.01)

    def plan(*args):
        value = plan_type(*args)
        plans.append((value, (value.steps, value.storage_buffers, value.storage_bytes)))
        return value

    monkeypatch.setattr(training, "Plan", plan)
    history = om.fit(model, values, targets, loss=loss, update=update, epochs=3, batch_size=3, seed=17, verbose=0)
    rng = np.random.default_rng(17)

    for epoch in range(1, 4):
        total = 0.0

        for x, y in om.batches(values, targets, batch_size=3, shuffle=True, rng=rng):
            error = x @ expected - y
            total += len(x) * np.mean(error**2)
            expected -= 0.01 * (x.reshape(-1, 2).T @ (2 * error.reshape(-1, 1) / error.size))

        assert history[epoch] == pytest.approx(total / len(values))

    np.testing.assert_allclose(weight.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert model.parameters()[0] is weight and model.training
    assert [x.shape[0] for x, _ in prepared] == [3, 1]
    assert len(plans) == len(updates) == 2

    for value, counters in plans:
        assert (value.steps, value.storage_buffers, value.storage_bytes) == counters

def test_fit_preserves_interacting_point_higher_order_objectives(api_device):
    class Cubic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = om.tensor(1.5, om.float64, api_device)
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
        return nn.mse_loss(second, labels)

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

        assert history[epoch] == pytest.approx(total / len(values))

    assert model.weight.item() == pytest.approx(weight)

def test_fit_full_and_tail_plans_share_batchnorm_and_dropout_state(api_device):
    def model():
        return nn.Sequential([
            nn.BatchNorm(2, dtype=om.float64, device=api_device),
            nn.Dropout(0.25, seed=7),
            nn.Linear(2, 1, dtype=om.float64, device=api_device, seed=8),
        ])

    actual, reference = model(), model()
    actual.eval()
    values = np.sin(np.arange(30)).reshape(5, 1, 3, 2)
    labels = np.cos(np.arange(15)).reshape(5, 1, 3, 1)
    history = om.fit(
        actual, values, labels, loss=lambda x, y: nn.mse_loss(actual(x), y),
        update=lambda p, g: om.sgd(p, g, 0.01), epochs=2, batch_size=2, shuffle=False, verbose=0,
    )
    prepared = {}

    for epoch in range(1, 3):
        total = 0.0

        for x, y in om.batches(values, labels, batch_size=2):
            size = len(x)

            if size not in prepared:
                inputs = om.tensor(x, om.float64, api_device)
                targets = om.tensor(y, om.float64, api_device)
                loss = nn.mse_loss(reference(inputs), targets)
                parameters = reference.parameters()
                gradients = om.grad(loss, parameters)
                plan = om.Plan([loss], [(p, om.sgd(p, g, 0.01)) for p, g in zip(parameters, gradients, strict=True)])
                prepared[size] = (inputs, targets, loss, plan)

            inputs, targets, loss, plan = prepared[size]
            inputs.copy_from(x)
            targets.copy_from(y)
            plan.run()
            total += size * loss.item()

        assert history[epoch] == pytest.approx(total / len(values))

    assert actual.training and all(child.training for child in actual.children())

    for name, value in actual.state().items():
        np.testing.assert_array_equal(value.numpy(), reference.state()[name].numpy())

@pytest.mark.parametrize("integer", [int, np.int64])
def test_fit_generator_and_seed_reproduce_epoch_orders(api_device, integer):
    values = np.sin(np.arange(12)).reshape(6, 2).astype(np.float32)
    models = [nn.Linear(2, 1, seed=9, device=api_device) for _ in range(2)]
    rng = np.random.default_rng(11)
    state = deepcopy(rng.bit_generator.state)
    histories = []

    for model, options in zip(models, [{"seed": integer(11)}, {"rng": rng}], strict=True):
        histories.append(om.fit(
            model, values, loss=lambda x: (model(x) * model(x)).mean(),
            update=lambda p, g: om.sgd(p, g, 0.01),
            epochs=integer(3), batch_size=integer(2), verbose=0, **options,
        ))

    assert histories[0] == histories[1]

    for lhs, rhs in zip(models[0].parameters(), models[1].parameters(), strict=True):
        np.testing.assert_array_equal(lhs.numpy(), rhs.numpy())

    assert rng.bit_generator.state != state
    reference = np.random.default_rng(11)

    for _ in range(3):
        reference.permutation(len(values))

    assert rng.bit_generator.state == reference.bit_generator.state

def test_fit_preserves_supported_input_dtypes_without_implicit_casts(api_device):
    model = nn.Linear(1, 1, dtype=om.float64, device=api_device, seed=7)
    dtypes = [(np.float16, om.float16), (np.float32, om.float32), (np.float64, om.float64),
              (np.int32, om.int32), (np.bool_, om.bool_)]
    arrays = [np.ones((3, 1), dtype=dtype) for dtype, _ in dtypes]
    received = []

    def loss(*inputs):
        received.append(inputs)
        assert [x.dtype for x in inputs] == [dtype for _, dtype in dtypes]
        assert all(x.device == model.weight.device for x in inputs)
        return model(inputs[2]).mean()

    om.fit(model, *arrays, loss=loss, update=lambda p, g: om.sgd(p, g, 0.01), batch_size=2, verbose=0)
    assert len(received) == 2

@pytest.mark.parametrize("options,error", [
    ({"epochs": 0}, ValueError), ({"epochs": -1}, ValueError),
    ({"epochs": True}, TypeError), ({"epochs": 1.5}, TypeError),
    ({"report_every": 0}, ValueError), ({"report_every": False}, TypeError),
    ({"verbose": 3}, ValueError), ({"verbose": -1}, ValueError),
    ({"verbose": True}, TypeError), ({"verbose": "quiet"}, TypeError),
    ({"seed": 7, "rng": np.random.default_rng(7)}, ValueError),
    ({"shuffle": False, "seed": 7}, ValueError),
    ({"batch_size": 0}, ValueError), ({"batch_size": True}, TypeError),
])
def test_fit_rejects_invalid_options_before_changing_mode_or_state(options, error):
    model = nn.Linear(1, 1, seed=7)
    model.eval()
    before = model.weight.numpy()
    rng = options.get("rng")
    state = deepcopy(rng.bit_generator.state) if rng else None

    with pytest.raises(error):
        om.fit(model, np.zeros((3, 1), np.float32), loss=lambda x: model(x).mean(),
               update=lambda p, g: om.sgd(p, g, 0.01), **options)

    assert not model.training
    np.testing.assert_array_equal(model.weight.numpy(), before)

    if rng:
        assert rng.bit_generator.state == state

@pytest.mark.parametrize("values,options,error", [
    (np.zeros((0, 1), np.float32), {}, ValueError),
    (np.zeros((1, 1), np.float32), {"drop_last": True}, ValueError),
    (np.zeros((3, 1), np.int64), {}, TypeError),
    (np.zeros((3, 1), dtype=">f8"), {}, TypeError),
    (np.zeros((), np.float32), {}, ValueError),
    ([1, 2, 3], {}, TypeError),
])
def test_fit_rejects_unsupported_training_data(values, options, error):
    model = nn.Linear(1, 1, seed=7)

    with pytest.raises(error):
        om.fit(model, values, loss=lambda x: model(x).mean(),
               update=lambda p, g: om.sgd(p, g, 0.01), **options)

@pytest.mark.parametrize("arrays,options,error,message", [
    ((), {"loss": None, "epochs": 0}, TypeError, "loss and update must be callable"),
    ((), {"epochs": 0, "verbose": 3}, ValueError, "epochs must be positive"),
    ((), {"verbose": 3, "batch_size": 0}, ValueError, "verbose must be 0, 1, or 2"),
    ((), {"batch_size": 0}, ValueError, "batches requires at least one NumPy array"),
    ((np.zeros((3, 1), np.int64),), {"batch_size": 0}, ValueError, "batch_size must be positive"),
    ((None,), {"shuffle": False, "seed": 7, "rng": np.random.default_rng(7)}, ValueError,
     "Set only one of seed and rng"),
    ((None,), {"shuffle": True, "seed": -1}, ValueError, "seed must be non-negative"),
    ((np.zeros((2, 1), np.int64), np.zeros((3, 1))), {}, ValueError,
     "All arrays must have the same number of samples"),
    ((np.zeros((0, 1), np.int64),), {}, ValueError,
     "Training requires at least one batch. Check sample count and drop_last."),
    ((np.zeros((1, 1), np.int64),), {"batch_size": 2, "drop_last": True}, ValueError,
     "Training requires at least one batch. Check sample count and drop_last."),
])
def test_fit_preserves_first_error_before_callbacks_or_state_changes(arrays, options, error, message):
    model = nn.Linear(1, 1, seed=7)
    model.eval()
    before = {name: value.numpy() for name, value in model.state().items()}
    rng = options.get("rng")
    state = deepcopy(rng.bit_generator.state) if isinstance(rng, np.random.Generator) else None

    def unexpected(*args):
        pytest.fail("Invalid training arguments reached a callback")

    kwargs = {"loss": unexpected, "update": unexpected, "verbose": 0, **options}

    with pytest.raises(error) as caught:
        om.fit(model, *arrays, **kwargs)

    assert str(caught.value) == message
    assert not model.training

    for name, value in model.state().items():
        np.testing.assert_array_equal(value.numpy(), before[name])

    if state is not None:
        assert rng.bit_generator.state == state

def test_fit_requires_a_model_parameters_data_and_graph_builders():
    model = nn.Linear(1, 1, seed=7)
    values = np.zeros((3, 1), np.float32)

    with pytest.raises(TypeError, match="Module"):
        om.fit(model.weight, values, loss=lambda x: x.mean(), update=lambda p, g: p)

    with pytest.raises(ValueError, match="registered parameter"):
        om.fit(nn.Tanh(), values, loss=lambda x: x.mean(), update=lambda p, g: p)

    with pytest.raises(ValueError, match="at least one"):
        om.fit(model, loss=lambda x: x.mean(), update=lambda p, g: p)

    with pytest.raises(TypeError, match="callable"):
        om.fit(model, values, loss=None, update=lambda p, g: p)

    with pytest.raises(TypeError, match="callable"):
        om.fit(model, values, loss=lambda x: x.mean(), update=None)

@pytest.mark.parametrize("loss,update,error", [
    (lambda x: 1.0, lambda p, g: p, TypeError),
    (lambda x: x, lambda p, g: p, ValueError),
    (lambda x: x.mean().astype(om.int32), lambda p, g: p, ValueError),
    (lambda x: x.mean(), lambda p, g: 1.0, TypeError),
])
def test_fit_validates_callback_outputs(loss, update, error):
    model = nn.Linear(1, 1, seed=7)

    with pytest.raises(error):
        om.fit(model, np.zeros((3, 1), np.float32), loss=loss, update=update, verbose=0)

@pytest.mark.parametrize("failure", ["loss", "update"])
def test_fit_keeps_completed_updates_when_tail_graph_construction_fails(failure):
    model = nn.Linear(1, 1, bias=False, dtype=om.float64, seed=7)
    model.weight.copy_from(np.array([[0.5]]))
    values = np.arange(1., 6.).reshape(5, 1)
    calls = []
    size = 0

    def loss(x):
        nonlocal size
        size = x.shape[0]
        calls.append(("loss", size))

        if size == 1 and failure == "loss":
            raise RuntimeError("tail loss failed")

        output = model(x)
        objective = (output * output).mean()
        assert not objective.realized
        return objective

    def update(parameter, gradient):
        calls.append(("update", size))
        assert not gradient.realized

        if size == 1 and failure == "update":
            raise RuntimeError("tail update failed")

        return om.sgd(parameter, gradient, 0.01)

    with pytest.raises(RuntimeError, match=f"tail {failure} failed"):
        om.fit(model, values, loss=loss, update=update, batch_size=2, shuffle=False, verbose=0)

    expected_calls = [("loss", 2), ("update", 2), ("loss", 1)]

    if failure == "update":
        expected_calls.append(("update", 1))

    assert calls == expected_calls
    assert model.weight.item() == pytest.approx(0.5 * (1 - 0.02 * 2.5) * (1 - 0.02 * 12.5))

class Terminal(io.StringIO):
    def isatty(self):
        return True

def test_fit_reports_loss_without_extra_tensor_reads(api_device, monkeypatch, capsys):
    stream = Terminal()
    bars = []
    reads = []
    item = om.Tensor.item

    def bar(*args, **kwargs):
        value = tqdm(*args, file=stream, mininterval=0, ascii=True, **kwargs)
        bars.append(value)
        return value

    def read(self):
        reads.append(self)
        return item(self)

    monkeypatch.setattr(tqdm_auto, "tqdm", bar)
    monkeypatch.setattr(om.Tensor, "item", read)
    values = np.sin(np.arange(10)).reshape(5, 2).astype(np.float32)
    models = [nn.Linear(2, 1, seed=9, device=api_device) for _ in range(3)]
    histories = []

    for verbose, model in zip((1, 2, 0), models, strict=True):
        reads.clear()
        stream.seek(0)
        stream.truncate()
        histories.append(om.fit(
            model, values, loss=lambda x: model(x).mean(), update=lambda p, g: om.sgd(p, g, 0.01),
            epochs=5, batch_size=2, shuffle=False, report_every=3, verbose=verbose,
        ))
        assert list(histories[-1]) == [1, 3, 5]
        assert len(reads) == 9

        if verbose == 1:
            assert bars[-1].n == 15
            assert all(text in stream.getvalue() for text in ("Epoch 5/5", "100%", "loss="))
            assert capsys.readouterr().out == ""
        else:
            assert stream.getvalue() == ""
            assert capsys.readouterr().out.count("epoch ") == (3 if verbose == 2 else 0)

    assert histories[0] == histories[1] == histories[2]

    for left, right in zip(models[0].parameters(), models[2].parameters(), strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())

def test_fit_closes_progress_and_propagates_callback_errors(monkeypatch):
    stream = Terminal()
    bars = []

    def bar(*args, **kwargs):
        value = tqdm(*args, file=stream, **kwargs)
        bars.append(value)
        return value

    def fail(*args):
        raise RuntimeError("training callback failed")

    monkeypatch.setattr(tqdm_auto, "tqdm", bar)
    model = nn.Linear(1, 1, seed=7)

    with pytest.raises(RuntimeError, match="training callback failed"):
        om.fit(model, np.zeros((3, 1), np.float32), loss=fail, update=lambda p, g: p)

    assert len(bars) == 1 and bars[0].disable
