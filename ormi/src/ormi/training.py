"""Train models with reusable Plans."""

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
    """Train a model with reusable batch Plans.

    Parameters
    ----------
    model : ormi.nn.Module
        Model with at least one registered parameter. Parameters and buffers
        must share a device. Batches are copied there.
    *arrays : numpy.ndarray
        Arrays with matching lengths along axis 0. Supported dtypes are
        float16, float32, float64, int32, and bool in the machine's byte order.
        Dtypes stay unchanged.
    loss : callable
        Called as ``loss(*inputs)`` with one Tensor per array. Build the loss
        using ``model`` and return a Tensor with one floating-point value.
    update : callable
        Called as ``update(parameter, gradient)`` per registered parameter.
        Return its next value as a Tensor expression without modifying it.
    epochs : int, default=1
        Positive number of passes over the data.
    batch_size : int, default=32
        Positive maximum samples per batch.
    shuffle : bool, default=True
        Shuffle all arrays together at the start of each epoch.
    seed : int or None, default=None
        Nonnegative seed for reproducible shuffling across the whole call.
        Each epoch advances the generator. Requires ``shuffle=True`` and ``rng=None``.
    rng : numpy.random.Generator or None, default=None
        Generator advanced for shuffling. Requires ``shuffle=True``.
    drop_last : bool, default=False
        Discard an incomplete final batch. Otherwise, train on it without padding.
    report_every : int, default=1
        Report epochs divisible by this positive interval, plus the first and
        final epochs.
    verbose : {0, 1, 2}, default=1
        0: silent. 1: terminal/notebook progress bar, plain logs when redirected.
        2: plain epoch logs.

    Returns
    -------
    history : dict[int, float]
        Reported losses keyed by epoch, starting at 1. Values use
        ``sum(batch_size * batch_loss) / sum(batch_size)``, using each batch's
        loss before its updates.

    Raises
    ------
    TypeError
        If an input or option type is invalid, a dtype is unsupported, or a
        callback returns a non-Tensor.
    ValueError
        If no batch or parameter exists, model devices differ, options are
        incompatible or out of range, or loss is not one floating-point value.

    Notes
    -----
    Callbacks run once per batch shape. Batches of that shape reuse input
    tensors and the Plan. Gradients stay lazy and support higher-order derivatives.

    Only reported epochs read losses with ``Tensor.item()``. Progress adds no
    tensor reads. Losses are weighted by batch size even when they sum values
    or ignore targets.

    The model enters training mode and stays there. Do not add or replace its
    parameters, buffers, or layers during training. Keep its mode and input
    shapes and dtypes fixed. Failed batches do not undo earlier updates.
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
