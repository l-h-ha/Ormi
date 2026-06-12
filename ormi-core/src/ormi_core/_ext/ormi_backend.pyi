"""
Ormi C++ Backend
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ADD', 'CONST', 'COS', 'CPU', 'CUDA', 'DIV', 'DataBuffer', 'Device', 'DeviceEnum', 'EXP', 'EXPAND', 'LEAF', 'LOG', 'MATMUL', 'MUL', 'NEG', 'Node', 'OpEnum', 'POW', 'RESHAPE', 'SIN', 'SUB', 'SUM', 'TRANSPOSE', 'TensorView', 'add', 'buffer_to_numpy', 'compute_gradients', 'cos', 'div', 'execute_graph', 'exp', 'expand', 'matmul', 'mul', 'neg', 'numpy_to_buffer', 'reshape', 'sin', 'sub', 'sum', 'transpose', 'zero_grad_graph']
class DataBuffer:
    @property
    def bytes(self) -> int:
        ...
    @property
    def shape(self) -> list[int]:
        ...
class Device:
    type: DeviceEnum
    def __init__(self, type: DeviceEnum = ..., id: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def is_type(self, arg0: str) -> bool:
        ...
    @property
    def id(self) -> int:
        ...
    @id.setter
    def id(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class DeviceEnum:
    """
    Members:
    
      CPU
    
      CUDA
    """
    CPU: typing.ClassVar[DeviceEnum]  # value = <DeviceEnum.CPU: 0>
    CUDA: typing.ClassVar[DeviceEnum]  # value = <DeviceEnum.CUDA: 1>
    __members__: typing.ClassVar[dict[str, DeviceEnum]]  # value = {'CPU': <DeviceEnum.CPU: 0>, 'CUDA': <DeviceEnum.CUDA: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Node:
    data: DataBuffer
    grad: Node
    view: TensorView
    @typing.overload
    def __init__(self, arg0: DataBuffer, arg1: bool, arg2: Device) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OpEnum, arg1: collections.abc.Sequence[Node], arg2: Device) -> None:
        ...
    @property
    def is_realized(self) -> bool:
        ...
    @property
    def op(self) -> OpEnum:
        ...
    @property
    def requires_grad(self) -> bool:
        ...
    @property
    def shape(self) -> list[int]:
        ...
    @property
    def strides(self) -> list[int]:
        ...
class OpEnum:
    """
    Members:
    
      LEAF
    
      CONST
    
      ADD
    
      MUL
    
      SUB
    
      DIV
    
      POW
    
      MATMUL
    
      NEG
    
      EXP
    
      LOG
    
      SIN
    
      COS
    
      SUM
    
      TRANSPOSE
    
      RESHAPE
    
      EXPAND
    """
    ADD: typing.ClassVar[OpEnum]  # value = <OpEnum.ADD: 7>
    CONST: typing.ClassVar[OpEnum]  # value = <OpEnum.CONST: 1>
    COS: typing.ClassVar[OpEnum]  # value = <OpEnum.COS: 6>
    DIV: typing.ClassVar[OpEnum]  # value = <OpEnum.DIV: 10>
    EXP: typing.ClassVar[OpEnum]  # value = <OpEnum.EXP: 3>
    EXPAND: typing.ClassVar[OpEnum]  # value = <OpEnum.EXPAND: 16>
    LEAF: typing.ClassVar[OpEnum]  # value = <OpEnum.LEAF: 0>
    LOG: typing.ClassVar[OpEnum]  # value = <OpEnum.LOG: 4>
    MATMUL: typing.ClassVar[OpEnum]  # value = <OpEnum.MATMUL: 12>
    MUL: typing.ClassVar[OpEnum]  # value = <OpEnum.MUL: 9>
    NEG: typing.ClassVar[OpEnum]  # value = <OpEnum.NEG: 2>
    POW: typing.ClassVar[OpEnum]  # value = <OpEnum.POW: 11>
    RESHAPE: typing.ClassVar[OpEnum]  # value = <OpEnum.RESHAPE: 15>
    SIN: typing.ClassVar[OpEnum]  # value = <OpEnum.SIN: 5>
    SUB: typing.ClassVar[OpEnum]  # value = <OpEnum.SUB: 8>
    SUM: typing.ClassVar[OpEnum]  # value = <OpEnum.SUM: 13>
    TRANSPOSE: typing.ClassVar[OpEnum]  # value = <OpEnum.TRANSPOSE: 14>
    __members__: typing.ClassVar[dict[str, OpEnum]]  # value = {'LEAF': <OpEnum.LEAF: 0>, 'CONST': <OpEnum.CONST: 1>, 'ADD': <OpEnum.ADD: 7>, 'MUL': <OpEnum.MUL: 9>, 'SUB': <OpEnum.SUB: 8>, 'DIV': <OpEnum.DIV: 10>, 'POW': <OpEnum.POW: 11>, 'MATMUL': <OpEnum.MATMUL: 12>, 'NEG': <OpEnum.NEG: 2>, 'EXP': <OpEnum.EXP: 3>, 'LOG': <OpEnum.LOG: 4>, 'SIN': <OpEnum.SIN: 5>, 'COS': <OpEnum.COS: 6>, 'SUM': <OpEnum.SUM: 13>, 'TRANSPOSE': <OpEnum.TRANSPOSE: 14>, 'RESHAPE': <OpEnum.RESHAPE: 15>, 'EXPAND': <OpEnum.EXPAND: 16>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TensorView:
    def is_contiguous(self) -> bool:
        ...
    @property
    def shape(self) -> list[int]:
        ...
    @shape.setter
    def shape(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def strides(self) -> list[int]:
        ...
    @strides.setter
    def strides(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
def add(arg0: Node, arg1: Node) -> Node:
    ...
def buffer_to_numpy(arg0: DataBuffer) -> numpy.typing.NDArray[numpy.float32]:
    ...
def compute_gradients(arg0: Node) -> None:
    ...
def cos(arg0: Node) -> Node:
    ...
def div(arg0: Node, arg1: Node) -> Node:
    ...
def execute_graph(arg0: Node) -> None:
    ...
def exp(arg0: Node) -> Node:
    ...
def expand(arg0: Node, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> Node:
    ...
def matmul(arg0: Node, arg1: Node) -> Node:
    ...
def mul(arg0: Node, arg1: Node) -> Node:
    ...
def neg(arg0: Node) -> Node:
    ...
def numpy_to_buffer(arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], arg1: DeviceEnum) -> DataBuffer:
    ...
def reshape(arg0: Node, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> Node:
    ...
def sin(arg0: Node) -> Node:
    ...
def sub(arg0: Node, arg1: Node) -> Node:
    ...
def sum(arg0: Node, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], arg2: bool) -> Node:
    ...
def transpose(arg0: Node, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> Node:
    ...
def zero_grad_graph(arg0: Node) -> None:
    ...
ADD: OpEnum  # value = <OpEnum.ADD: 7>
CONST: OpEnum  # value = <OpEnum.CONST: 1>
COS: OpEnum  # value = <OpEnum.COS: 6>
CPU: DeviceEnum  # value = <DeviceEnum.CPU: 0>
CUDA: DeviceEnum  # value = <DeviceEnum.CUDA: 1>
DIV: OpEnum  # value = <OpEnum.DIV: 10>
EXP: OpEnum  # value = <OpEnum.EXP: 3>
EXPAND: OpEnum  # value = <OpEnum.EXPAND: 16>
LEAF: OpEnum  # value = <OpEnum.LEAF: 0>
LOG: OpEnum  # value = <OpEnum.LOG: 4>
MATMUL: OpEnum  # value = <OpEnum.MATMUL: 12>
MUL: OpEnum  # value = <OpEnum.MUL: 9>
NEG: OpEnum  # value = <OpEnum.NEG: 2>
POW: OpEnum  # value = <OpEnum.POW: 11>
RESHAPE: OpEnum  # value = <OpEnum.RESHAPE: 15>
SIN: OpEnum  # value = <OpEnum.SIN: 5>
SUB: OpEnum  # value = <OpEnum.SUB: 8>
SUM: OpEnum  # value = <OpEnum.SUM: 13>
TRANSPOSE: OpEnum  # value = <OpEnum.TRANSPOSE: 14>
