from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest

import ormi as om
import ormi.training as training
from ormi import nn

###
###
###

@pytest.mark.parametrize("shape", [(7,), (7, 1), (7, 2, 1, 3), (7, 0, 2)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.bool_])
def test_batches_preserve_shapes_dtypes_alignment_and_tail(shape, dtype):
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    labels = np.arange(7, dtype=np.int32)
    weights = np.linspace(0, 1, 7)
    result = list(om.batches(values, labels, weights, batch_size=3))
    assert [len(batch[0]) for batch in result] == [3, 3, 1]

    for start, (batch, target, weight) in zip(range(0, 7, 3), result, strict=True):
        assert batch.shape == (min(3, 7 - start), *shape[1:])
        assert batch.dtype == values.dtype and target.dtype == labels.dtype
        np.testing.assert_array_equal(batch, values[start:start + 3])
        np.testing.assert_array_equal(target, labels[start:start + 3])
        np.testing.assert_array_equal(weight, weights[start:start + 3])

    np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]), values)

def test_batches_return_tuples_and_unshuffled_views_without_consuming_inputs():
    values = np.arange(48).reshape(8, 6)[::-1, ::2]
    values.setflags(write=False)
    iterator = om.batches(values, batch_size=np.int64(3))
    assert iter(iterator) is iterator
    result = list(iterator)
    assert all(isinstance(batch, tuple) and len(batch) == 1 for batch in result)
    assert all(np.shares_memory(batch[0], values) and not batch[0].flags.writeable for batch in result)
    np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]), values)
    assert list(iterator) == []

@pytest.mark.parametrize("count,size,drop,lengths", [
    (0, 3, False, []), (0, 3, True, []), (2, 3, False, [2]), (2, 3, True, []),
    (6, 3, False, [3, 3]), (6, 3, True, [3, 3]), (7, 3, False, [3, 3, 1]), (7, 3, True, [3, 3]),
])
def test_final_batch_policy_is_explicit(count, size, drop, lengths):
    values = np.arange(count)
    result = list(om.batches(values, batch_size=size, drop_last=drop))
    assert [len(batch[0]) for batch in result] == lengths

    if lengths:
        np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]), values[:sum(lengths)])

def test_shuffled_batches_match_one_paired_permutation_and_do_not_mutate_sources():
    values = np.arange(39, dtype=np.float32).reshape(13, 1, 3)
    labels = np.arange(13, dtype=np.int32)
    original = values.copy()
    expected = np.random.default_rng(7).permutation(13)
    result = list(om.batches(values, labels, batch_size=4, shuffle=True, seed=7))
    np.testing.assert_array_equal(np.concatenate([batch[1] for batch in result]), expected)

    for batch, target in result:
        np.testing.assert_array_equal(batch, values[target])
        assert batch.dtype == values.dtype and target.dtype == labels.dtype
        assert not np.shares_memory(batch, values)

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(labels, np.arange(13))
    repeated = list(om.batches(labels, batch_size=4, shuffle=True, seed=np.int64(7)))
    np.testing.assert_array_equal(np.concatenate([batch[0] for batch in repeated]), expected)
    dropped = list(om.batches(labels, batch_size=4, shuffle=True, seed=7, drop_last=True))
    np.testing.assert_array_equal(np.concatenate([batch[0] for batch in dropped]), expected[:12])

def test_shared_generator_advances_once_per_epoch_and_replays_the_same_sequence():
    values = np.arange(19)
    first = np.random.default_rng(23)
    repeated = np.random.default_rng(23)
    reference = np.random.default_rng(23)
    orders = []

    for _ in range(3):
        before = deepcopy(first.bit_generator.state)
        iterator = om.batches(values, batch_size=4, shuffle=True, rng=first)
        np.testing.assert_equal(first.bit_generator.state, before)
        order = np.concatenate([batch[0] for batch in iterator])
        same = np.concatenate([batch[0] for batch in om.batches(values, batch_size=4, shuffle=True, rng=repeated)])
        np.testing.assert_array_equal(order, same)
        np.testing.assert_array_equal(order, reference.permutation(19))
        np.testing.assert_equal(first.bit_generator.state, reference.bit_generator.state)
        orders.append(order)

    assert not np.array_equal(orders[0], orders[1])

def test_empty_iterations_and_unshuffled_batches_leave_rng_state_untouched():
    rng = np.random.default_rng(7)
    before = deepcopy(rng.bit_generator.state)
    assert list(om.batches(np.zeros((0, 3)), batch_size=2, shuffle=True, rng=rng)) == []
    assert list(om.batches(np.zeros((1, 3)), batch_size=2, shuffle=True, rng=rng, drop_last=True)) == []
    np.testing.assert_equal(rng.bit_generator.state, before)
    global_state = np.random.get_state()
    list(om.batches(np.arange(10), batch_size=3))
    list(om.batches(np.arange(10), batch_size=3, shuffle=True, seed=7))
    np.testing.assert_equal(np.random.get_state(), global_state)

@pytest.mark.parametrize("kwargs,error", [
    ({"batch_size": 0}, ValueError), ({"batch_size": -2}, ValueError),
    ({"batch_size": True}, TypeError), ({"batch_size": np.bool_(True)}, TypeError),
    ({"batch_size": 1.5}, TypeError), ({"batch_size": "2"}, TypeError),
    ({"batch_size": 2, "shuffle": 1}, TypeError), ({"batch_size": 2, "drop_last": "yes"}, TypeError),
    ({"batch_size": 2, "shuffle": True, "seed": -1}, ValueError),
    ({"batch_size": 2, "shuffle": True, "seed": 1.5}, TypeError),
    ({"batch_size": 2, "shuffle": True, "seed": True}, TypeError),
    ({"batch_size": 2, "seed": 7}, ValueError),
    ({"batch_size": 2, "rng": np.random.default_rng(7)}, ValueError),
    ({"batch_size": 2, "shuffle": True, "rng": np.random.RandomState(7)}, TypeError),
    ({"batch_size": 2, "shuffle": True, "seed": 7, "rng": np.random.default_rng(7)}, ValueError),
])
def test_batches_validate_options_at_call_time(kwargs, error):
    with pytest.raises(error):
        om.batches(np.arange(5), **kwargs)

@pytest.mark.parametrize("arrays,options,error,message", [
    ((), {"batch_size": 0}, ValueError, "batches requires at least one NumPy array"),
    ((None,), {"batch_size": True, "shuffle": 1}, TypeError, "batch_size must be a non-boolean integer"),
    ((None,), {"batch_size": 1.5, "shuffle": 1}, TypeError, "batch_size must be an integer"),
    ((None,), {"batch_size": 0, "shuffle": 1}, ValueError, "batch_size must be positive"),
    ((None,), {"shuffle": 1, "seed": 7, "rng": np.random.default_rng(7)}, TypeError,
     "shuffle and drop_last must be booleans"),
    ((None,), {"seed": 7, "rng": np.random.default_rng(7)}, ValueError, "Set only one of seed and rng"),
    ((None,), {"rng": object()}, ValueError, "seed and rng require shuffle=True"),
    ((None,), {"shuffle": True, "rng": object()}, TypeError, "rng must be a NumPy Generator"),
    ((None,), {"shuffle": True, "seed": True}, TypeError, "seed must be a non-boolean integer"),
    ((None,), {"shuffle": True, "seed": 1.5}, TypeError, "seed must be an integer"),
    ((None,), {"shuffle": True, "seed": -1}, ValueError, "seed must be non-negative"),
    ((np.zeros(2), np.zeros(3), np.array(0)), {}, ValueError, "Input 2 must have a leading sample axis"),
    ((np.zeros(2), np.zeros(3)), {}, ValueError, "All arrays must have the same number of samples"),
])
def test_batches_preserve_first_error_without_advancing_rng(arrays, options, error, message):
    kwargs = {"batch_size": 2, **options}
    rng = kwargs.get("rng")
    state = deepcopy(rng.bit_generator.state) if isinstance(rng, np.random.Generator) else None

    with pytest.raises(error) as caught:
        om.batches(*arrays, **kwargs)

    assert str(caught.value) == message

    if state is not None:
        assert rng.bit_generator.state == state

@pytest.mark.parametrize("count,drop", [(0, False), (1, True), (5, False)])
def test_seeded_batches_create_rng_only_when_iteration_has_data(count, drop, monkeypatch):
    create_rng = np.random.default_rng
    calls = []

    def record(seed):
        calls.append(seed)
        return create_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", record)
    iterator = om.batches(np.arange(count), batch_size=2, shuffle=True, seed=np.int64(7), drop_last=drop)
    assert calls == []
    result = list(iterator)

    if result:
        assert calls == [7]
        expected = create_rng(7).permutation(count)
        np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]), expected)
    else:
        assert calls == []

def test_batches_reject_invalid_inputs_without_realizing_tensors():
    with pytest.raises(ValueError, match="at least one"):
        om.batches(batch_size=2)

    with pytest.raises(ValueError, match="same number"):
        om.batches(np.zeros((3, 2)), np.zeros(4), batch_size=2)

    with pytest.raises(ValueError, match="sample axis"):
        om.batches(np.array(1), batch_size=2)

    for value in [None, [1, 2], om.tensor([1, 2]).sin()]:
        with pytest.raises(TypeError, match="NumPy array"):
            om.batches(value, batch_size=2)

        if isinstance(value, om.Tensor):
            assert not value.realized

def test_memmapped_arrays_can_be_batched_without_loading_a_copy(tmp_path):
    values = np.memmap(tmp_path / "data.bin", mode="w+", shape=(7, 2, 3), dtype=np.float32)
    values[:] = np.arange(42).reshape(7, 2, 3)
    result = list(om.batches(values, batch_size=3))
    assert all(np.shares_memory(batch[0], values) for batch in result)
    np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]), values)

def test_full_and_tail_plans_share_parameters_and_match_numpy_sgd(api_device):
    values = np.sin(np.arange(42)).reshape(7, 1, 3, 2)
    targets = np.cos(np.arange(21)).reshape(7, 1, 3, 1)
    weights = np.array([[0.2], [-0.1]])
    weight = om.tensor(weights, om.float64, api_device)
    prepared = {}
    rng = np.random.default_rng(7)
    runs = {3: 0, 1: 0}

    for _ in range(2):
        for batch, labels in om.batches(values, targets, batch_size=3, shuffle=True, rng=rng):
            size = len(batch)

            if size not in prepared:
                inputs = om.tensor(np.zeros_like(batch), om.float64, api_device)
                target = om.tensor(np.zeros_like(labels), om.float64, api_device)
                loss = nn.mse_loss(inputs @ weight, target)
                gradient = om.grad(loss, weight)
                curvature = om.grad(gradient.sum(), weight)
                plan = om.Plan([loss, gradient, curvature], [(weight, om.sgd(weight, gradient, 0.01))])
                assert not loss.realized and not gradient.realized and not curvature.realized
                counters = (plan.steps, plan.storage_buffers, plan.storage_bytes)
                prepared[size] = (inputs, target, loss, gradient, curvature, plan, counters)

            inputs, target, loss, gradient, curvature, plan, counters = prepared[size]
            inputs.copy_from(batch)
            target.copy_from(labels)
            error = batch @ weights - labels
            expected_gradient = batch.reshape(-1, 2).T @ (2 * error.reshape(-1, 1) / error.size)
            plan.run()
            assert loss.item() == pytest.approx(np.mean(error**2))
            np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=1e-12, atol=1e-12)
            flat = batch.reshape(-1, 2)
            expected_curvature = 2 * flat.T @ (flat @ np.ones_like(weights)) / error.size
            np.testing.assert_allclose(curvature.numpy(), expected_curvature, rtol=1e-12, atol=1e-12)
            weights -= 0.01 * expected_gradient
            np.testing.assert_allclose(weight.numpy(), weights, rtol=1e-12, atol=1e-12)
            assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == counters
            runs[size] += 1

    assert len(prepared) == 2 and runs == {3: 4, 1: 2}

def test_full_and_tail_batchnorm_plans_preserve_state_and_cuda_replay(api_device):
    values = np.sin(np.arange(30)).reshape(5, 1, 3, 2)
    model = nn.BatchNorm(2, affine=False, momentum=0.5, dtype=om.float64, device=api_device)
    prepared = {}
    replays = {}
    running_mean = np.zeros(2)
    running_var = np.ones(2)
    rng = np.random.default_rng(7)

    for _ in range(2):
        for (batch,) in om.batches(values, batch_size=2, shuffle=True, rng=rng):
            size = len(batch)

            if size not in prepared:
                inputs = om.tensor(np.zeros_like(batch), om.float64, api_device)
                output = model(inputs)
                plan = om.Plan([output])
                prepared[size] = (inputs, output, plan)

            inputs, output, plan = prepared[size]
            inputs.copy_from(batch)

            if size in replays:
                replays[size].run()
            else:
                plan.run()

                if api_device != "cpu":
                    replays[size] = plan.record()

            flat = batch.reshape(-1, 2)
            expected = (batch - flat.mean(0)) / np.sqrt(flat.var(0) + 1e-5)
            running_mean = 0.5 * running_mean + 0.5 * flat.mean(0)
            running_var = 0.5 * running_var + 0.5 * flat.var(0, ddof=1)
            np.testing.assert_allclose(output.numpy(), expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(model.running_mean.numpy(), running_mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(model.running_var.numpy(), running_var, rtol=1e-12, atol=1e-12)

    assert len(prepared) == 2

@pytest.mark.parametrize("samples,size,drop,sizes", [(17, 8, False, [8, 1]), (17, 8, True, [8]), (3, 8, False, [3])])
def test_example_builds_only_one_plan_per_compatible_batch_shape(api_device, samples, size, drop, sizes, monkeypatch):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "minibatch.py"))
    plan = om.Plan
    created = []

    def prepare(*args, **kwargs):
        value = plan(*args, **kwargs)
        created.append(value)
        return value

    monkeypatch.setattr(training, "Plan", prepare)
    model = example["train"](
        2, api_device, samples=samples, batch_size=size, drop_last=drop, progress=False,
    )
    assert isinstance(model, nn.Module)
    assert len(created) == len(sizes)
