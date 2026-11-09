"""Train lazy graphs with reusable batch inputs and Plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ormi_core import DType, Plan, Tensor, bool_, float16, float32, float64, grad, int32, nn

from .data import _validate_batches, batches

__all__ = ["fit"]

###
###
###

def fit(
    model: nn.Module,
    *arrays: NDArray[Any],
    loss: Callable[..., Tensor],
    update: Callable[[Tensor, Tensor], Tensor],
    epochs: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    drop_last: bool = False,
    report_every: int = 1,
    verbose: Literal[0, 1, 2] = 1,
) -> dict[int, float]:
    """Train a model using batches and reusable Plans.

    Parameters
    ----------
    model : ormi.nn.Module
        Model with at least one registered parameter. All parameters and
        buffers must use the same device. Batches are copied to that device.
    *arrays : numpy.ndarray
        NumPy arrays with the same length along axis 0. Supported dtypes are
        float16, float32, float64, int32, and bool in the machine's byte order.
        Dtypes stay unchanged.
    loss : callable
        Called as ``loss(*inputs)`` with one Tensor per array. Build the
        loss using ``model`` and return one floating-point value as a Tensor.
        This callback runs only when building the graph for a new batch shape.
    update : callable
        Called as ``update(parameter, gradient)`` for each registered parameter.
        Return the next value as a Tensor expression. Leave the parameter unchanged.
    epochs : int, default=1
        Number of passes over the data. Must be positive.
    batch_size : int, default=32
        Maximum samples per batch. Must be positive.
    shuffle : bool, default=True
        Shuffle all arrays together at the start of each epoch.
    seed : int or None, default=None
        Nonnegative seed for reproducible shuffling across the whole call.
        Each epoch advances the generator. Requires ``shuffle=True`` and cannot
        be used with ``rng``.
    rng : numpy.random.Generator or None, default=None
        External generator to advance for shuffling. Requires ``shuffle=True``.
    drop_last : bool, default=False
        Discard an incomplete final batch. Otherwise, train on it without padding.
    report_every : int, default=1
        Report epochs divisible by this positive interval, plus the first and
        final epochs.
    verbose : {0, 1, 2}, default=1
        Use 0 for no output, 1 for a terminal or notebook progress bar with
        plain logs when redirected, or 2 for plain epoch logs.

    Returns
    -------
    history : dict[int, float]
        Reported losses keyed by epoch, starting at 1. Each value is
        ``sum(batch_size * batch_loss) / sum(batch_size)``, using the loss
        computed before each batch's updates.

    Raises
    ------
    TypeError
        If inputs or options have invalid types, a dtype is unsupported, or a
        callback does not return a Tensor.
    ValueError
        If no training batch or registered parameter exists, model devices
        differ, options are incompatible or out of range, or the loss
        is not one floating-point value.

    Notes
    -----
    The callbacks run once per batch shape. Later batches of that shape
    reuse the input tensors and Plan. Gradients remain lazy and
    can include higher-order derivatives.

    Only reported epochs read losses with ``Tensor.item()``. The progress bar
    adds no other tensor reads. The same weights apply when the loss sums
    values or ignores targets.

    The model is switched to training mode and left there. Do not add or replace
    its parameters, buffers, or layers during training. Keep its mode and input
    shapes and dtypes fixed. If a batch fails, earlier updates are not undone.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an ORMI Module")

    if not callable(loss) or not callable(update):
        raise TypeError("loss and update must be callable")

    for name, value in (("epochs", epochs), ("report_every", report_every)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")

        if value < 1:
            raise ValueError(f"{name} must be positive")

    epochs = int(epochs)
    report_every = int(report_every)

    if isinstance(verbose, (bool, np.bool_)) or not isinstance(verbose, (int, np.integer)):
        raise TypeError("verbose must be 0, 1, or 2")

    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2")

    # Validate data and shuffle options before changing mode or drawing randomness.
    batch_size, seed, count = _validate_batches(arrays, batch_size, shuffle, seed, rng, drop_last)
    batch_count = count // batch_size if drop_last else (count + batch_size - 1) // batch_size

    if batch_count == 0:
        raise ValueError("Training requires at least one batch. Check sample count and drop_last.")

    supported = {
        np.dtype("float16"): float16,
        np.dtype("float32"): float32,
        np.dtype("float64"): float64,
        np.dtype("int32"): int32,
        np.dtype("bool"): bool_,
    }
    dtypes: list[DType] = []

    for index, array in enumerate(arrays):
        dtype = supported.get(array.dtype)

        if dtype is None:
            raise TypeError(f"Input {index} has unsupported dtype {array.dtype}. Convert it explicitly.")

        dtypes.append(dtype)

    parameters = model.parameters()

    if not parameters:
        raise ValueError("Training requires at least one registered parameter")

    device = parameters[0].device

    if any(value.device != device for value in [*parameters, *model.buffers()]):
        raise ValueError("Model parameters and buffers must use one device")

    from tqdm.auto import tqdm

    generator = rng

    if shuffle and generator is None:
        generator = np.random.default_rng(seed)

    prepared: dict[int, tuple[tuple[Tensor, ...], Tensor, Plan]] = {}
    history: dict[int, float] = {}
    model.train()

    with tqdm(
        total=epochs * batch_count, desc="Training", unit="batch", dynamic_ncols=True,
        disable=None if verbose == 1 else True,
    ) as progress:
        show_bar = not progress.disable

        for epoch in range(1, epochs + 1):
            report = epoch == 1 or epoch % report_every == 0 or epoch == epochs
            total_loss = 0.0
            seen = 0

            if show_bar:
                progress.set_description(f"Epoch {epoch}/{epochs}", refresh=False)

                if not report:
                    progress.set_postfix({}, refresh=False)

            for batch in batches(
                *arrays, batch_size=batch_size, shuffle=shuffle, rng=generator, drop_last=drop_last,
            ):
                size = len(batch[0])
                entry = prepared.get(size)

                if entry is None:
                    # Run earlier shapes before building another graph with shared module buffers.
                    inputs = tuple(
                        Tensor(array, dtype, device)
                        for array, dtype in zip(batch, dtypes, strict=True)
                    )
                    objective, plan = _build_plan(inputs, parameters, loss, update)
                    prepared[size] = (inputs, objective, plan)
                else:
                    inputs, objective, plan = entry

                    for destination, source in zip(inputs, batch, strict=True):
                        destination.copy_from(source)

                plan.run()

                if report:
                    total_loss += size * float(objective.item())
                    seen += size

                    if show_bar:
                        progress.set_postfix(loss=f"{total_loss / seen:.6g}", refresh=False)

                if show_bar:
                    progress.update()

            if report:
                history[epoch] = total_loss / seen

                if verbose and not show_bar:
                    print(f"epoch {epoch:5d}  loss {history[epoch]:.8e}")

    return history

def _build_plan(
    inputs: tuple[Tensor, ...],
    parameters: list[Tensor],
    loss: Callable[..., Tensor],
    update: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, Plan]:
    objective = loss(*inputs)

    if not isinstance(objective, Tensor):
        raise TypeError("loss must return an ORMI Tensor")

    if objective.numel != 1 or objective.dtype not in (float16, float32, float64):
        raise ValueError("loss must return one floating-point value")

    gradients = grad(objective, parameters)
    updates: list[tuple[Tensor, Tensor]] = []

    for parameter, gradient in zip(parameters, gradients, strict=True):
        value = update(parameter, gradient)

        if not isinstance(value, Tensor):
            raise TypeError("update must return an ORMI Tensor")

        updates.append((parameter, value))

    return objective, Plan([objective], updates)
