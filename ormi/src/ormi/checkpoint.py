"""Save model tensors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.lib.npyio import NpzFile

from ormi_core import Tensor, bool_, float16, float32, float64, int32
from ormi_core.nn import Module

###
###
###

__all__ = ["save", "load"]

_DTYPES = {
    float16: np.dtype("float16"),
    float32: np.dtype("float32"),
    float64: np.dtype("float64"),
    int32: np.dtype("int32"),
    bool_: np.dtype("bool"),
}

def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}

    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate checkpoint metadata key: {name}")

        result[name] = value

    return result

def _read_metadata(archive: NpzFile) -> dict[str, str]:
    if len(archive.files) != len(set(archive.files)) or "__ormi__" not in archive.files:
        raise ValueError("Checkpoint has duplicate entries or no format metadata")

    metadata = archive["__ormi__"]

    if metadata.dtype != np.uint8 or metadata.ndim != 1:
        raise ValueError("Invalid checkpoint metadata")

    manifest = json.loads(metadata.tobytes(), object_pairs_hook=_unique_object)

    if not isinstance(manifest, dict) or set(manifest) != {"version", "state"}:
        raise ValueError("Invalid checkpoint metadata")

    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("Unsupported checkpoint version")

    names = manifest["state"]

    if not isinstance(names, dict) or any(not isinstance(key, str) for key in names.values()):
        raise ValueError("Invalid checkpoint state names")

    return names

###
###
###

def save(model: Module, path: str | os.PathLike[str]) -> None:
    """Save named parameters and buffers.

    Parameters
    ----------
    model : Module
        Model whose registered state will be saved. State tensors must be
        contiguous. Shared tensors are stored once.
    path : str or os.PathLike[str]
        Destination file. The parent directory must exist. An existing file
        is replaced only after the new archive has been written successfully.
        No filename extension is added.

    Raises
    ------
    TypeError
        If ``model`` is not a Module.
    ValueError
        If a registered state tensor is not contiguous.
    OSError
        If the file cannot be written or replaced.

    Notes
    -----
    The file contains NumPy arrays and JSON metadata, without pickle.
    GPU data is copied to the CPU. Do not change model state
    while saving.

    Model structure, training mode, optimizer and random-generator state,
    graphs, and Plans are not saved. Create the model before loading.
    """

    if not isinstance(model, Module):
        raise TypeError("save() expects a Module.")

    ###

    arrays: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}
    shared: dict[int, str] = {}

    for name, tensor in model.state().items():
        if not tensor.is_contiguous():
            raise ValueError(f"Model state tensor must be contiguous: {name}")

        key = shared.get(id(tensor))

        if key is None:
            key = f"tensor_{len(arrays)}"
            arrays[key] = tensor.numpy()
            shared[id(tensor)] = key

        names[name] = key

    manifest = json.dumps({"version": 1, "state": names}, ensure_ascii=True).encode("utf-8")
    arrays["__ormi__"] = np.frombuffer(manifest, dtype=np.uint8)
    destination = Path(path)
    _write_archive(destination, arrays)

def _write_archive(destination: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write and sync the archive before replacing its destination."""

    temporary: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, destination)
        temporary = None

    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

def load(model: Module, path: str | os.PathLike[str]) -> None:
    """Check saved state, then copy it into the model's existing tensors.

    Parameters
    ----------
    model : Module
        Existing model with matching state names, shapes, dtypes, and shared
        tensors. Registered state tensors must be contiguous.
    path : str or os.PathLike[str]
        Archive previously written by ``save``.

    Raises
    ------
    TypeError
        If ``model`` is not a Module.
    ValueError
        If archive contents or model state fail validation.
    OSError
        If the file cannot be opened or read.

    Notes
    -----
    All arrays are checked before copying. Loading updates tensor values in
    place, preserving the Tensor objects and their storage. Copies add no
    gradient relationship. Rerun the Plan or replay to refresh computed outputs.

    Model structure, training mode, optimizer and random-generator state,
    graphs, and Plans are not restored. Do not run or modify the model while
    loading. If a device copy fails, earlier copies are not undone.
    """

    if not isinstance(model, Module):
        raise TypeError("load expects an ORMI Module")

    archive = np.load(path, allow_pickle=False)

    if not isinstance(archive, NpzFile):
        raise ValueError("Checkpoint must be a NumPy .npz archive")

    copies: list[tuple[Tensor, np.ndarray]] = []

    with archive:
        names = _read_metadata(archive)
        state = model.state()

        if state.keys() != names.keys():
            missing = sorted(state.keys() - names.keys())
            unexpected = sorted(names.keys() - state.keys())
            raise ValueError(f"Checkpoint names do not match: missing={missing}, unexpected={unexpected}")

        keys = set(names.values())
        expected_keys = {f"tensor_{i}" for i in range(len(keys))}

        if keys != expected_keys or set(archive.files) != keys | {"__ormi__"}:
            raise ValueError("Checkpoint tensor entries do not match its metadata")

        shared: dict[int, str] = {}
        used_keys: set[str] = set()

        for name, tensor in state.items():
            key = names[name]
            identity = id(tensor)

            if identity in shared:
                if shared[identity] != key:
                    raise ValueError(f"Checkpoint tensor sharing does not match: {name}")

                continue

            if key in used_keys:
                raise ValueError(f"Checkpoint tensor sharing does not match: {name}")

            if not tensor.is_contiguous():
                raise ValueError(f"Checkpoint state must be contiguous: {name}")

            array = archive[key]

            if array.shape != tensor.shape:
                raise ValueError(f"Checkpoint shape does not match: {name}")

            if array.dtype != _DTYPES[tensor.dtype]:
                raise ValueError(f"Checkpoint dtype does not match: {name}")

            shared[identity] = key
            used_keys.add(key)
            copies.append((tensor, array))

    # Every entry has been read and validated.
    for tensor, array in copies:
        tensor.copy_from(array)
