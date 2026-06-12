from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np

from .exceptions import raise_device_mismatch

from ._ext import ormi_backend
from ._ext.ormi_backend import OpEnum, DeviceEnum, Device, Node

class Tensor:
    def __init__(
        self, 
        data: list | float | ArrayLike | None = None,
        device: str | DeviceEnum | Device | None = None,
        requires_grad: bool = False,
        _node: Node | None = None
    ):
        # Device
        if device is None:
            self.device = Device(DeviceEnum.CPU, 0)
        elif isinstance(device, str):
            dev_str = device.lower()
            if dev_str == "cuda":
                self.device = Device(DeviceEnum.CUDA, 0)
            elif dev_str == "cpu":
                self.device = Device(DeviceEnum.CPU, 0)
            else:
                raise_device_mismatch(msg="Unsupported string backend", got=device)
        elif isinstance(device, DeviceEnum):
            self.device = Device(device, 0)
        elif isinstance(device, Device):
            self.device = device
        else:
            raise_device_mismatch(
                msg="Device argument is of invalid type.",
                got=type(device)
            )

        # Init
        if _node is not None:
            # Intermediate node
            self._node = _node
        elif data is not None:
            # Leaf node
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            buffer_ptr = ormi_backend.numpy_to_buffer(data, self.device.type)
            self._node = Node(buffer_ptr, requires_grad, self.device)
        else:
            self._node = None

    ###
    ###
    ###

    @property
    def op(self) -> OpEnum:
        return self._node.op if self._node else OpEnum.LEAF

    @property
    def requires_grad(self) -> bool:
        return self._node.requires_grad if self._node else False

    @property
    def shape(self) -> tuple | None:
        if self._node:
            return tuple(self._node.shape)
        return None
    
    @property
    def grad(self) -> Tensor | None:
        if self._node and self._node.grad:
            return Tensor(_node=self._node.grad, device=self.device)
        return None

    ###
    ###
    ###

    def _ensure_tensor(self, other) -> Tensor:
        if isinstance(other, Tensor):
            if self.device.type != other.device.type:
                raise_device_mismatch(msg="Cannot mix devices in operation.")
            return other
        return Tensor(data=other, device=self.device)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.ADD, [self._node, other._node], self.device)
        return Tensor(_node=new_node, device=self.device)
    
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.SUB, [self._node, other._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.MUL, [self._node, other._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.DIV, [self._node, other._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __pow__(self, power):
        p_node = self._ensure_tensor(power)
        new_node = Node(OpEnum.POW, [self._node, p_node._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matmul requires both operands to be Tensors.")
        if self.device.type != other.device.type:
            raise_device_mismatch(msg="Cannot perform MATMUL across different devices")

        new_node = Node(OpEnum.MATMUL, [self._node, other._node], self.device)
        return Tensor(_node=new_node, device=self.device)
    
    def __neg__(self):
        new_node = Node(OpEnum.NEG, [self._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.SUB, [other._node, self._node], self.device)
        return Tensor(_node=new_node, device=self.device)

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        new_node = Node(OpEnum.DIV, [other._node, self._node], self.device)
        return Tensor(_node=new_node, device=self.device)
    
    ###
    ### View
    ###

    def reshape(self, shape: tuple | list) -> Tensor:
        new_node = ormi_backend.reshape(self._node, list(shape))
        return Tensor(_node=new_node, device=self.device)

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        if axes is None:
            axes = tuple(reversed(range(len(self.shape))))
        new_node = ormi_backend.transpose(self._node, list(axes))
        return Tensor(_node=new_node, device=self.device)
    
    ###
    ###
    ###

    def realize(self) -> Tensor:
        if self._node and self._node.is_realized:
            return self
        ormi_backend.execute_graph(self._node)
        return self
        
    def numpy(self) -> np.ndarray:
        self.realize()
        if not self._node or not self._node.is_realized:
            raise RuntimeError("Cannot fetch an unrealized tensor.")

        raw_array = ormi_backend.buffer_to_numpy(self._node.data)
        byte_strides = tuple(s * 4 for s in self._node.view.strides)
        
        from numpy.lib.stride_tricks import as_strided
        return as_strided(raw_array, shape=self.shape, strides=byte_strides)

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients.")
        ormi_backend.compute_gradients(self._node)

    def zero_grad(self):
        if self._node:
            self._node.grad = None

    def zero_grad_graph(self):
        if self._node:
            ormi_backend.zero_grad_graph(self._node)

    def __repr__(self):
        state = "realized" if (self._node and self._node.is_realized) else "lazy"
        return f"Tensor(op={self.op.name}, shape={self.shape}, state={state}, device={self.device.type.name})"##