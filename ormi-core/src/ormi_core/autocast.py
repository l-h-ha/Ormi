from __future__ import annotations

from ._ext._C import Autocast, DType, float16

###
###
###

class autocast:
    def __init__(self, dtype: DType = float16):
        self.dtype = dtype
        self._scope: Autocast | None = None

    def __enter__(self) -> autocast:
        if self._scope is not None:
            raise RuntimeError("An autocast context cannot be entered twice.")

        self._scope = Autocast(self.dtype)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._scope = None

__all__ = ["autocast"]
