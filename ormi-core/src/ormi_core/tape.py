from __future__ import annotations

from collections.abc import Iterable

from ._ext import _C
from .tensor import Tensor


class ComputeTape:
    def __init__(self):
        self._tape = _C.ComputeTape()

    def capture(self, roots: Iterable[Tensor]) -> None:
        nodes = []

        for item in roots:
            if item is None:
                continue

            if not isinstance(item, Tensor):
                raise TypeError("Expected Tensor objects.")

            nodes.append(item._ensure_node())

        self._tape.capture(nodes)

    def execute(self) -> None:
        self._tape.execute()

    def dump(self) -> None:
        self._tape.dump()