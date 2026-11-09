import ormi as om
import numpy as np
from ormi.nn import Linear, Module
from ormi.ops import matmul
from ormi.optim import sgd

x = om.tensor([1.0, 2.0])
Linear("three", 2)  # error: reportCallIssue, reportArgumentType
matmul(x, "weight")  # error: reportArgumentType
sgd(x, x, "fast")  # error: reportArgumentType
x.reshape(3)  # error: reportArgumentType
om.diff(x, "input")  # error: reportArgumentType
om.Plan([1])  # error: reportArgumentType
om.partial(x, x, coordinate="x")  # error: reportArgumentType
om.partial(x, x, order=0.5)  # error: reportArgumentType
om.save(x, "model.npz")  # error: reportArgumentType
om.load(Linear(2, 2), x)  # error: reportArgumentType
Linear(2, 2).register_buffer(x, name=1)  # error: reportArgumentType
om.batches(x, batch_size=2)  # error: reportArgumentType
om.batches(batch_size="three")  # error: reportArgumentType
om.batches(batch_size=2, seed="seven")  # error: reportArgumentType
om.batches(batch_size=2, rng=7)  # error: reportArgumentType

values = np.zeros((3, 2), dtype=np.float32)
model = Linear(2, 1)
om.fit(x, values, loss=lambda x: x.mean(), update=lambda p, g: p)  # error: reportArgumentType
om.fit(model, x, loss=lambda x: x.mean(), update=lambda p, g: p)  # error: reportArgumentType
om.fit(model, values, loss=lambda x: 1.0, update=lambda p, g: p)  # error: reportArgumentType
om.fit(model, values, loss=lambda x: x.mean(), update=lambda p, g: 1.0)  # error: reportArgumentType
om.fit(model, values, loss=lambda x: x.mean(), update=lambda p, g: p, verbose=3)  # error: reportArgumentType

###
###
###

class InvalidModule(Module):
    def forward(self, input: om.Tensor) -> int:  # error: reportIncompatibleMethodOverride
        return 1

class InvalidLinear(Linear):  # error: reportGeneralTypeIssues
    pass
