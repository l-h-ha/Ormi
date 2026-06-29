from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import raise_device_mismatch
from ._ext import _C
from ._ext._C import Device, DeviceEnum, Node, OpEnum

def _normalize_device(device: str | DeviceEnum | Device | None) -> Device:
    if device is None:
        return _C.cpu()

    if isinstance(device, str):
        dev = device.lower()

        if dev == "cpu":
            return _C.cpu()
        if dev == "cuda":
            return _C.cuda()
        if dev.startswith("cuda:"):
            return _C.cuda(int(dev.split(":", 1)[1]))

        raise_device_mismatch(msg="Unsupported string device.", got=device)

    if isinstance(device, DeviceEnum):
        if device == _C.CPU:
            return _C.cpu()
        if device == _C.CUDA:
            return _C.cuda()
        raise_device_mismatch(msg="Unsupported DeviceEnum.", got=device)

    if isinstance(device, Device):
        return device

    raise_device_mismatch(
        msg="Device argument is of invalid type.",
        got=type(device),
    )

def _devices_match(a: Device, b: Device) -> bool:
    return a.type == b.type and a.id == b.id

class Tensor:
    def __init__(
        self,
        data: list | float | ArrayLike | None = None,
        device: str | DeviceEnum | Device | None = None,
        requires_grad: bool = False,
        _node: Node | None = None,
    ):
        if _node is not None:
            self._node = _node
            self.device = _node.device
            return

        self.device = _normalize_device(device)

        if data is None:
            self._node = None
            return

        array = np.asarray(data, dtype=np.float32)
        array = np.ascontiguousarray(array)

        buffer = _C.numpy_to_buffer(array, self.device)
        self._node = Node(buffer, requires_grad, "")

    @classmethod
    def from_node(cls, node: Node) -> Tensor:
        return cls(_node=node)

    @property
    def op(self) -> OpEnum:
        return self._node.op if self._node else _C.LEAF

    @property
    def requires_grad(self) -> bool:
        return self._node.requires_grad if self._node else False

    @property
    def shape(self) -> tuple[int, ...] | None:
        if not self._node:
            return None
        return tuple(self._node.shape)

    @property
    def dtype(self):
        return self._node.dtype if self._node else None

    @property
    def is_realized(self) -> bool:
        return bool(self._node and self._node.is_realized())

    def _ensure_node(self) -> Node:
        if self._node is None:
            raise RuntimeError("Tensor has no backing node.")
        return self._node

    def _ensure_tensor(self, other) -> Tensor:
        if isinstance(other, Tensor):
            if not _devices_match(self.device, other.device):
                raise_device_mismatch(msg="Cannot mix devices in operation.")
            return other

        return Tensor(data=other, device=self.device)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.add(self._ensure_node(), other._ensure_node()))

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.sub(self._ensure_node(), other._ensure_node()))

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.mul(self._ensure_node(), other._ensure_node()))

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.div(self._ensure_node(), other._ensure_node()))

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matmul requires both operands to be Tensors.")

        if not _devices_match(self.device, other.device):
            raise_device_mismatch(msg="Cannot perform matmul across different devices.")

        return Tensor.from_node(_C.ops.matmul(self._ensure_node(), other._ensure_node()))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.sub(other._ensure_node(), self._ensure_node()))

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return Tensor.from_node(_C.ops.div(other._ensure_node(), self._ensure_node()))

    def reshape(self, shape: tuple[int, ...] | list[int]) -> Tensor:
        return Tensor.from_node(_C.ops.reshape(self._ensure_node(), list(shape)))

    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> Tensor:
        if axes is None:
            if self.shape is None:
                raise RuntimeError("Cannot infer transpose axes for an empty Tensor.")
            axes = tuple(reversed(range(len(self.shape))))

        return Tensor.from_node(_C.ops.transpose(self._ensure_node(), list(axes)))

    def sum(self, keepdims: bool = False) -> Tensor:
        return Tensor.from_node(_C.ops.sum(self._ensure_node(), keepdims))

    def sum_axes(self, axes: Iterable[int], keepdims: bool = False) -> Tensor:
        return Tensor.from_node(_C.ops.sum_axes(self._ensure_node(), list(axes), keepdims))

    def realize(self) -> Tensor:
        self._ensure_node().realize()
        return self

    def numpy(self) -> np.ndarray:
        node = self._ensure_node()
        node.realize()

        if not node.is_realized():
            raise RuntimeError("Cannot fetch an unrealized tensor.")

        raw_array = _C.buffer_to_numpy(node.data)
        byte_strides = tuple(int(s) * 4 for s in node.view.strides)

        from numpy.lib.stride_tricks import as_strided
        return as_strided(
            raw_array,
            shape=self.shape,
            strides=byte_strides,
        )

    def grad(self, inputs: list[Tensor]) -> list[Tensor | None]:
        nodes = [item._ensure_node() for item in inputs]
        grads = _C.grad(self._ensure_node(), nodes)
        return [
            Tensor.from_node(g) if g is not None else None
            for g in grads
        ]

    def __repr__(self):
        state = "realized" if self.is_realized else "lazy"
        device = _C.device_to_string(self.device)
        return (
            f"Tensor(op={self.op.name}, "
            f"shape={self.shape}, "
            f"state={state}, "
            f"device={device})"
        )