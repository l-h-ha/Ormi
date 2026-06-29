from __future__ import annotations

from collections.abc import Sequence

from ._ext import _C
from .tensor import Tensor

class SGD:
    def __init__(
        self,
        params: Sequence[Tensor],
        grads: Sequence[Tensor | None],
        lr: float,
    ):
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length.")

        pairs = []

        for param, grad in zip(params, grads):
            if grad is None:
                continue

            if not isinstance(param, Tensor):
                raise TypeError("SGD params must be Tensor objects.")

            if not isinstance(grad, Tensor):
                raise TypeError("SGD grads must be Tensor objects or None.")

            pairs.append((param._ensure_node(), grad._ensure_node()))

        self._optimizer = _C.SGD(pairs, float(lr))

    def step(self) -> None:
        self._optimizer.step()