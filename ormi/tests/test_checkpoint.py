import importlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

import ormi as om
from ormi import nn

###
###
###

class State(nn.Module):
    def forward(self, input):
        return input

def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    manifest = json.loads(arrays["__ormi__"].tobytes())
    return arrays, manifest

def write_checkpoint(path, arrays, manifest):
    arrays["__ormi__"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)

    with path.open("wb") as stream:
        np.savez(stream, **arrays)

###
###
###

def test_checkpoint_restores_buffers_and_higher_order_graph_without_replacing_tensors(api_device, tmp_path):
    model = nn.Sequential([
        nn.Linear(2, 3, dtype=om.float64, device=api_device, seed=7),
        nn.BatchNorm(3, dtype=om.float64, device=api_device),
        nn.Tanh(),
        nn.Linear(3, 1, dtype=om.float64, device=api_device, seed=8),
    ])
    x = om.tensor(np.arange(12).reshape(2, 1, 3, 2) / 12, om.float64, api_device)
    model(x).numpy()
    model.eval()
    output = model(x)
    first = om.grad(output.sum(), x)
    second = om.grad(first.sum(), x)
    plan = om.Plan([output, first, second])
    assert not output.realized and not first.realized and not second.realized
    state = model.state()
    assert set(state) == {
        "0.weight", "0.bias", "1.weight", "1.bias", "1.running_mean", "1.running_var",
        "3.weight", "3.bias",
    }
    expected_state = {name: tensor.numpy() for name, tensor in state.items()}
    path = tmp_path / "model.npz"
    assert om.save(model, path) is None
    assert not output.realized and not first.realized and not second.realized
    plan.run()
    expected = [tensor.numpy() for tensor in [output, first, second]]
    counters = (plan.steps, plan.storage_buffers, plan.storage_bytes)

    for tensor in state.values():
        tensor.copy_from(np.full_like(tensor.numpy(), 0.2))

    plan.run()
    changed_output = output.numpy()
    assert not np.allclose(changed_output, expected[0])
    assert om.load(model, path) is None
    np.testing.assert_array_equal(output.numpy(), changed_output)

    for name, tensor in model.state().items():
        assert tensor is state[name]
        np.testing.assert_array_equal(tensor.numpy(), expected_state[name])

    plan.run()

    for tensor, reference in zip([output, first, second], expected, strict=True):
        np.testing.assert_allclose(tensor.numpy(), reference, rtol=1e-12, atol=1e-12)

    assert (plan.steps, plan.storage_buffers, plan.storage_bytes) == counters

def test_checkpoint_restores_a_prepared_training_plan_and_cuda_replay(api_device, tmp_path):
    model = nn.Linear(2, 1, dtype=om.float64, device=api_device, seed=4)
    x = om.tensor([[1., 2.], [-1., 3.]], om.float64, api_device)
    target = om.tensor([[0.5], [0.1]], om.float64, api_device)
    loss = nn.mse_loss(model(x), target)
    parameters = model.parameters()
    gradients = om.grad(loss, parameters)
    plan = om.Plan([loss], [
        (parameter, om.sgd(parameter, gradient, 0.01))
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ])
    path = tmp_path / "initial.npz"
    om.save(model, path)
    plan.run()
    expected_loss = loss.item()
    expected = [parameter.numpy() for parameter in parameters]
    om.load(model, path)
    plan.run()
    assert loss.item() == pytest.approx(expected_loss)

    for parameter, values in zip(parameters, expected, strict=True):
        np.testing.assert_allclose(parameter.numpy(), values, rtol=1e-12, atol=1e-12)

    if api_device != "cpu":
        replay = plan.record()
        del plan
        om.load(model, path)
        replay.run()
        assert loss.item() == pytest.approx(expected_loss)

        for parameter, values in zip(parameters, expected, strict=True):
            np.testing.assert_allclose(parameter.numpy(), values, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.bool_])
@pytest.mark.parametrize("shape", [(), (2, 1, 3), (2, 0, 3)])
def test_checkpoint_preserves_all_buffer_dtypes_and_shapes(api_device, tmp_path, dtype, shape):
    dtype_map = {
        np.float16: om.float16, np.float32: om.float32, np.float64: om.float64,
        np.int32: om.int32, np.bool_: om.bool_,
    }
    values = np.asarray(np.arange(np.prod(shape)).reshape(shape) + 1, dtype=dtype)
    source = State()
    source.register_buffer(om.tensor(values, dtype_map[dtype], api_device), "values")
    path = tmp_path / "state.npz"
    om.save(source, path)
    destination = State()
    buffer = om.tensor(np.zeros(shape, dtype=dtype), dtype_map[dtype], api_device)
    destination.register_buffer(buffer, "values")
    om.load(destination, path)
    np.testing.assert_array_equal(buffer.numpy(), values)

def test_checkpoint_shares_one_stored_array_per_tensor_and_supports_another_device(api_device, tmp_path):
    layer = nn.Linear(2, 2, dtype=om.float64, device=api_device, seed=6)
    source = nn.Sequential([layer, layer])
    state = source.state()
    assert state["0.weight"] is state["1.weight"] is layer.weight
    assert state["0.bias"] is state["1.bias"] is layer.bias
    path = tmp_path / "shared.npz"
    om.save(source, path)
    arrays, manifest = read_checkpoint(path)
    assert len(arrays) == 3
    assert manifest["state"]["0.weight"] == manifest["state"]["1.weight"]
    other = nn.Linear(2, 2, dtype=om.float64, seed=9)
    destination = nn.Sequential([other, other])
    om.load(destination, path)

    for name, tensor in destination.state().items():
        np.testing.assert_array_equal(tensor.numpy(), state[name].numpy())

    if api_device != "cpu":
        om.save(destination, path)
        layer.weight.copy_from(np.zeros((2, 2)))
        om.load(source, path)
        np.testing.assert_array_equal(layer.weight.numpy(), other.weight.numpy())

@pytest.mark.parametrize("source_tied,target_tied", [(True, False), (False, True)])
def test_checkpoint_rejects_different_tensor_sharing_before_writing(tmp_path, source_tied, target_tied):
    def model(tied):
        first = nn.Linear(2, 2, seed=5)
        return nn.Sequential([first, first if tied else nn.Linear(2, 2, seed=6)])

    source = model(source_tied)
    destination = model(target_tied)
    before = {name: tensor.numpy() for name, tensor in destination.state().items()}
    path = tmp_path / "shared.npz"
    om.save(source, path)

    with pytest.raises(ValueError, match="sharing"):
        om.load(destination, path)

    for name, tensor in destination.state().items():
        np.testing.assert_array_equal(tensor.numpy(), before[name])

@pytest.mark.parametrize("damage", [
    "shape", "dtype", "object", "version", "bool_version", "missing_name", "extra_name",
    "missing_array", "extra_array", "invalid_metadata", "invalid_names", "unknown_metadata",
    "malformed_json", "duplicate_json",
])
def test_checkpoint_validates_every_entry_before_writing(tmp_path, damage):
    model = nn.Linear(2, 2, dtype=om.float64, seed=7)
    before = {name: tensor.numpy() for name, tensor in model.state().items()}
    path = tmp_path / "bad.npz"
    om.save(model, path)
    arrays, manifest = read_checkpoint(path)

    for key in manifest["state"].values():
        arrays[key] = np.full_like(arrays[key], 99)

    last = manifest["state"]["weight"]

    if damage == "shape":
        arrays[last] = np.ones((1, 4))
    elif damage == "dtype":
        arrays[last] = arrays[last].astype(np.float32)
    elif damage == "object":
        arrays[last] = arrays[last].astype(object)
    elif damage == "version":
        manifest["version"] = 999
    elif damage == "bool_version":
        manifest["version"] = True
    elif damage == "missing_name":
        del manifest["state"]["weight"]
    elif damage == "extra_name":
        manifest["state"]["other"] = last
    elif damage == "missing_array":
        del arrays[last]
    elif damage == "extra_array":
        arrays["tensor_99"] = np.ones(1)
    elif damage == "invalid_names":
        manifest["state"]["weight"] = 1
    elif damage == "unknown_metadata":
        manifest["other"] = 1

    write_checkpoint(path, arrays, manifest)

    if damage == "invalid_metadata":
        arrays["__ormi__"] = np.array([1.0])
        np.savez(path, **arrays)

    if damage in {"malformed_json", "duplicate_json"}:
        text = b'{' if damage == "malformed_json" else b'{"version":999,"version":1,"state":{}}'
        arrays["__ormi__"] = np.frombuffer(text, dtype=np.uint8)
        np.savez(path, **arrays)

    with pytest.raises(ValueError):
        om.load(model, path)

    for name, tensor in model.state().items():
        np.testing.assert_array_equal(tensor.numpy(), before[name])

def test_checkpoint_handles_empty_models_and_rejects_non_archives(tmp_path):
    path = tmp_path / "empty.npz"
    model = nn.Sequential([])
    om.save(model, path)
    om.load(model, path)

    with path.open("wb") as stream:
        np.save(stream, np.zeros(1))

    with pytest.raises(ValueError, match=".npz archive"):
        om.load(model, path)

    with pytest.raises(TypeError, match="Module"):
        om.save(object(), path)

    with pytest.raises(TypeError, match="Module"):
        om.load(object(), path)

@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("operation,expected", [
    (None, ["create", "savez", "flush", "fsync", "close", "replace"]),
    ("create", ["create"]),
    ("savez", ["create", "savez", "close", "unlink"]),
    ("flush", ["create", "savez", "flush", "close", "unlink"]),
    ("fsync", ["create", "savez", "flush", "fsync", "close", "unlink"]),
    ("close", ["create", "savez", "flush", "fsync", "close", "unlink"]),
    ("replace", ["create", "savez", "flush", "fsync", "close", "replace", "unlink"]),
    ("cleanup", ["create", "savez", "flush", "fsync", "close", "replace", "unlink"]),
])
def test_save_preserves_write_order_and_failure_cleanup(tmp_path, monkeypatch, existing, operation, expected):
    checkpoint = importlib.import_module("ormi.checkpoint")
    path = tmp_path / "model.npz"
    model = nn.Linear(2, 2, seed=7)
    values = {name: tensor.numpy() for name, tensor in model.state().items()}
    original = b"previous checkpoint"

    if existing:
        path.write_bytes(original)

    events = []
    streams = []
    error = OSError("test write failure")
    create_file = checkpoint.tempfile.NamedTemporaryFile
    write_arrays = checkpoint.np.savez
    sync_file = checkpoint.os.fsync
    replace_file = checkpoint.os.replace
    unlink_file = Path.unlink

    class Output:
        def __init__(self, raw):
            self.raw = raw

        @property
        def name(self):
            return self.raw.name

        def __enter__(self):
            self.raw.__enter__()
            return self

        def __exit__(self, *args):
            try:
                return self.raw.__exit__(*args)
            finally:
                events.append("close")

                if operation == "close":
                    raise error

        def flush(self):
            events.append("flush")

            if operation == "flush":
                raise error

            self.raw.flush()

        def fileno(self):
            return self.raw.fileno()

    def create(**kwargs):
        events.append("create")
        assert kwargs == dict(dir=tmp_path, prefix=".model.npz.", suffix=".tmp", delete=False)

        if operation == "create":
            raise error

        stream = Output(create_file(**kwargs))
        streams.append(stream)
        return stream

    def write(stream, **arrays):
        events.append("savez")

        if operation == "savez":
            stream.raw.write(b"incomplete archive")
            raise error

        write_arrays(stream.raw, **arrays)

    def sync(descriptor):
        events.append("fsync")
        assert descriptor == streams[-1].raw.fileno()

        if operation == "fsync":
            raise error

        sync_file(descriptor)

    def replace(source, destination):
        events.append("replace")
        assert all(stream.raw.closed for stream in streams)
        assert source.parent == destination.parent == tmp_path

        if operation in ("replace", "cleanup"):
            raise error

        replace_file(source, destination)

    def unlink(temporary, *, missing_ok=False):
        events.append("unlink")
        assert all(stream.raw.closed for stream in streams)
        assert temporary.parent == tmp_path and missing_ok

        if operation == "cleanup":
            raise PermissionError("test cleanup failure")

        unlink_file(temporary, missing_ok=missing_ok)

    monkeypatch.setattr(checkpoint.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(checkpoint.np, "savez", write)
    monkeypatch.setattr(checkpoint.os, "fsync", sync)
    monkeypatch.setattr(checkpoint.os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)

    if operation is None:
        om.save(model, path)
        arrays, manifest = read_checkpoint(path)

        for name, array in values.items():
            np.testing.assert_array_equal(arrays[manifest["state"][name]], array)
    else:
        with pytest.raises(OSError) as caught:
            om.save(model, path)

        assert caught.value is error

        if existing:
            assert path.read_bytes() == original
        else:
            assert not path.exists()

    assert events == expected
    assert all(stream.raw.closed for stream in streams)
    temporary = list(tmp_path.glob(".model.npz.*.tmp"))
    assert len(temporary) == (1 if operation == "cleanup" else 0)

    for item in temporary:
        unlink_file(item)

    assert list(tmp_path.iterdir()) == ([path] if existing or operation is None else [])

    for name, tensor in model.state().items():
        np.testing.assert_array_equal(tensor.numpy(), values[name])

def test_checkpoint_rejects_duplicate_zip_entries(tmp_path):
    model = nn.Linear(2, 2, seed=7)
    path = tmp_path / "duplicate.npz"
    om.save(model, path)

    with zipfile.ZipFile(path, "a") as archive:
        data = archive.read("tensor_0.npy")

        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("tensor_0.npy", data)

    with pytest.raises(ValueError, match="duplicate"):
        om.load(model, path)

def test_checkpoint_rejects_invalid_metadata_before_reading_model_state(tmp_path):
    class UnreadState(State):
        def state(self):
            raise AssertionError("Model state was read before checking metadata")

    path = tmp_path / "invalid.npz"
    metadata = json.dumps({"version": True, "state": {}}).encode()
    np.savez(path, __ormi__=np.frombuffer(metadata, dtype=np.uint8))

    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        om.load(UnreadState(), path)

@pytest.mark.parametrize("dtype,integer", [
    (np.float16, np.uint16), (np.float32, np.uint32), (np.float64, np.uint64),
])
def test_checkpoint_preserves_nonfinite_values_and_signed_zero_bits(api_device, tmp_path, dtype, integer):
    model = State()
    values = np.array([0., -0., np.inf, -np.inf, np.nan], dtype=dtype)
    kinds = {np.float16: om.float16, np.float32: om.float32, np.float64: om.float64}
    buffer = om.tensor(values, kinds[dtype], api_device)
    model.register_buffer(buffer, "values")
    path = tmp_path / "bits.npz"
    om.save(model, path)
    buffer.copy_from(np.zeros_like(values))
    om.load(model, path)
    np.testing.assert_array_equal(buffer.numpy().view(integer), values.view(integer))
