"""Batch host arrays."""

from __future__ import annotations

from collections.abc import Iterator
import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["batches"]

###
###
###

def batches(
    *arrays: NDArray[Any],
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    drop_last: bool = False,
) -> Iterator[tuple[NDArray[Any], ...]]:
    """Yield aligned batches along axis 0.

    Parameters
    ----------
    *arrays : numpy.ndarray
        One or more arrays with the same length along axis 0. Other dimensions
        and dtypes stay unchanged. Scalars and non-array inputs are rejected.
    batch_size : int
        Maximum number of samples per batch. Must be positive.
    shuffle : bool, default=False
        Shuffle all arrays using the same sample order.
    seed : int or None, default=None
        Nonnegative seed for a local generator. Separate calls with the same
        seed repeat the order. Requires ``shuffle=True`` and cannot be used with
        ``rng``.
    rng : numpy.random.Generator or None, default=None
        Generator to advance when iteration starts. Reuse it for reproducible
        but different orders across epochs. Requires ``shuffle=True``.
    drop_last : bool, default=False
        Discard an incomplete final batch. Otherwise, yield it without padding.

    Yields
    ------
    batch : tuple of numpy.ndarray
        One batch per input, in a tuple even for a single input. Without
        shuffling, batches share the source data. Shuffling returns copies.

    Raises
    ------
    TypeError
        If an input is not a NumPy array or an option has an invalid type.
    ValueError
        If no arrays are supplied, sample counts differ, an array is scalar,
        or batch and shuffle options are incompatible or out of range.

    Notes
    -----
    Arguments are checked on this call. The returned iterator can be used once.
    No tensors are created or copied to a device. Empty iterations draw no
    randomness. NumPy's global random state is unchanged. Do not change input
    shapes or dtypes during iteration.
    """

    batch_size, seed, count = _validate_batches(arrays, batch_size, shuffle, seed, rng, drop_last)
    return _iterate_batches(arrays, batch_size, count, shuffle, seed, rng, drop_last)

def _validate_batches(
    arrays: tuple[NDArray[Any], ...],
    batch_size: int,
    shuffle: bool,
    seed: int | None,
    rng: np.random.Generator | None,
    drop_last: bool,
) -> tuple[int, int | None, int]:
    """Check batch arguments and return batch size, seed, and sample count."""

    if not arrays:
        raise ValueError("batches requires at least one NumPy array")

    if isinstance(batch_size, (bool, np.bool_)):
        raise TypeError("batch_size must be a non-boolean integer")

    try:
        batch_size = operator.index(batch_size)
    except TypeError:
        raise TypeError("batch_size must be an integer") from None

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
        raise TypeError("shuffle and drop_last must be booleans")

    if seed is not None and rng is not None:
        raise ValueError("Set only one of seed and rng")

    if not shuffle and (seed is not None or rng is not None):
        raise ValueError("seed and rng require shuffle=True")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")

    if seed is not None:
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("seed must be a non-boolean integer")

        try:
            seed = operator.index(seed)
        except TypeError:
            raise TypeError("seed must be an integer") from None

        if seed < 0:
            raise ValueError("seed must be non-negative")

    for index, array in enumerate(arrays):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input {index} must be a NumPy array. Convert tensors explicitly.")

        if array.ndim == 0:
            raise ValueError(f"Input {index} must have a leading sample axis")

    count = arrays[0].shape[0]

    for array in arrays[1:]:
        if array.shape[0] != count:
            raise ValueError("All arrays must have the same number of samples")

    return batch_size, seed, count

def _iterate_batches(
    arrays: tuple[NDArray[Any], ...],
    batch_size: int,
    count: int,
    shuffle: bool,
    seed: int | None,
    rng: np.random.Generator | None,
    drop_last: bool,
) -> Iterator[tuple[NDArray[Any], ...]]:
    end = count - count % batch_size if drop_last else count

    if end == 0:
        return

    order = None

    if shuffle:
        generator = rng if rng is not None else np.random.default_rng(seed)
        order = generator.permutation(count)

    for start in range(0, end, batch_size):
        stop = min(start + batch_size, end)

        if order is None:
            yield tuple(array[start:stop] for array in arrays)
        else:
            indices = order[start:stop]
            yield tuple(array[indices] for array in arrays)
