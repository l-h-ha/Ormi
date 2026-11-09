# Neural Networks

The examples use `ormi_core`. `ormi.nn` exposes the same objects.
See [Python training](../../ormi/README.md#training) for `fit`, batching, and checkpoints.

Layers build tensor graphs for later execution. `grad` builds their derivative graphs.

```python
import numpy as np
import ormi_core as oc
from ormi_core import nn

model = nn.Sequential([
    nn.Linear(1, 32, seed=7),
    nn.Sin(),
    nn.Linear(32, 1, seed=8),
])
x = oc.Tensor([[-0.5], [0.5]])
prediction = model(x)
loss = nn.mse_loss(prediction, oc.zeros_like(prediction))
parameters = model.parameters()
gradients = oc.grad(loss, parameters)
updates = [
    (parameter, oc.sgd(parameter, gradient, 0.01))
    for parameter, gradient in zip(parameters, gradients, strict=True)
]
plan = oc.Plan([loss], updates)
plan.run()
```

For minibatches, keep the same input Tensor objects and call
`input.copy_from(batch)` before each `plan.run()`. Copying requires a realized,
contiguous leaf destination and an exact shape/dtype match. NumPy arrays and
realized contiguous tensors are accepted. Copies add no autograd edges and do
not rerun existing outputs. See the [minibatch example](../../ormi/examples/minibatch.py)
for inputs, labels, shuffling, and Plans reused by batch shape.

`Linear(in_features, out_features, bias=True, dtype=oc.float32,
device="cpu", seed=None)` accepts inputs shaped `[..., in_features]`. It
stores weights as `[in_features, out_features]` and bias as `[out_features]`,
so `forward` is `input @ weight + bias` without a transpose node.
Feature counts must be positive. Leading batch dimensions can be empty.
Initialization draws weights and biases uniformly from
`[-1/sqrt(in_features), 1/sqrt(in_features))` before conversion to the parameter
dtype. An explicit seed repeats the draws with the same C++ standard-library
implementation. Float16, Float32, and Float64 parameters are supported.
Initialization runs on the CPU, then copies values to the selected device.

`Linear(weight, bias=None)` uses supplied realized floating-point leaves without
copying them. `parameters()` lists the module's own parameters, then each child's
parameters recursively, in registration order. Shared tensors appear once.
Regression and binary losses require matching prediction and target shapes.
Integer differentiation and implicit loss broadcasting are not supported.

`model.state()` returns live named parameter and buffer references, including all
paths to shared tensors. Built-in modules name their state. Custom modules can
pass `name` to `register_parameter`, `register_buffer`, or `register_module`.
The Python `ormi.save` and `ormi.load` functions store and restore these values.
See [model checkpoints](../../ormi/README.md#model-checkpoints) for compatibility rules.

## Activations And Losses

Activations have function and layer forms, such as `nn.gelu(x)` and
`nn.GELU()(x)`: `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`,
`Softplus`, `Softsign`, `Mish`, `Softmax`, and `LogSoftmax`. `Sin` remains
available for sine networks. GELU uses the exact erf formula. Softplus uses
beta 1. LeakyReLU defaults to a nonnegative slope of 0.01. ReLU's derivative
at zero is 0. LeakyReLU uses its negative slope there. These rules do not make
the functions smooth at their breakpoints.

| Loss | Inputs and options |
| --- | --- |
| `mse_loss`, `l1_loss` | Matching floating-point predictions and targets. |
| `huber_loss`, `smooth_l1_loss` | Positive `delta` or nonnegative `beta`, both defaulting to 1. Smooth L1 with beta 0 is L1. |
| `binary_cross_entropy_with_logits` | Raw logits with targets in [0, 1]. Accepts an optional broadcastable `pos_weight`. No sigmoid call is needed. |
| `binary_cross_entropy` | Probabilities and targets in [0, 1]. Predictions are clipped by `eps=1e-7`, with representable bounds strictly between 0 and 1. |
| `cross_entropy` | Raw logits with Int32 class indices or matching floating-point target distributions. Options include `axis`, `ignore_index`, and `label_smoothing`. No softmax call is needed. |
| `nll_loss` | Log probabilities with Int32 class indices. Options include `axis` and `ignore_index`. |
| `kl_div` | Log probabilities and matching target probabilities, or log targets with `log_target=True`. |

Losses accept `reduction="none"`, `"sum"`, or `"mean"` and default to mean.
KL divergence defaults to `"batchmean"`, which sums all terms and
divides by the size of axis 0. `"mean"` averages all loss elements.
Cross-entropy and NLL mean exclude ignored indices, which default
to -100. A batch with no valid targets has sum 0 and mean NaN.

Softmax, log-softmax, cross-entropy, and NLL default to the last class axis.
All other axes are retained. Index targets have the logits shape with the
class axis removed. Soft targets have exactly the logits shape. Non-ignored
indices outside [0, number of classes) produce NaN, without an out-of-bounds
read or a host synchronization. Probability targets must be valid
distributions. Shape and dtype checks do not validate their numerical values.
Inputs to normalization and logits losses should be finite.

Float16 composite activations calculate in Float32 and return Float16.
Losses promote Float16 predictions and targets to Float32. Prediction/target
dtypes determine the result dtype, including promotion to Float64.
BCE `pos_weight` is converted to that dtype. All functions build lazy graphs,
including their derivatives, and work with prepared Plans and CUDA recording.

## Normalization

`nn.LayerNorm(normalized_shape)` and `nn.layer_norm(...)` normalize the matching
trailing axes using population variance and `eps=1e-5`. For input
`[B, ..., D1, ..., Dk]`, use `normalized_shape=(D1, ..., Dk)`. A single integer
normalizes the last axis. Normalized axis sizes must be positive. Leading batch
axes may be empty. Epsilon must be finite, positive, and representable in the
dtype used for statistics. Float16 uses Float32 statistics. Float32 and Float64
keep their precision, and the output keeps the input dtype. Optional weight and bias
must have exactly `normalized_shape`, use the input device, and match its dtype.
Float32 weight and bias are also accepted for Float16 inputs. Module weights
start at one and biases at zero. Set `bias=False` or `elementwise_affine=False`
to omit them. The function also accepts lazy weight and bias expressions.

`nn.BatchNorm(C, axis=-1)` normalizes every axis except the channel axis. It
accepts `[B, ..., C]` by default. Use `axis=1` for `[B, C, ...]`. Training uses
population variance for normalization and unbiased variance for the running
estimate. The default update is `running = 0.9 * running + 0.1 * batch`. `momentum`
must be a finite number in `[0, 1]`. Cumulative averaging is unsupported. Nonempty training
inputs need at least two values per channel. Empty inputs leave statistics
unchanged. `affine=False` omits weights and biases. `track_running_stats=False`
uses batch statistics in both modes. Float16 uses Float32 statistics and running
buffers. Float32 and Float64 keep their precision. Float16 also accepts a layer
whose parameters and buffers are Float32.

## Dropout

`nn.Dropout(p=0.5, seed=None)` and `nn.dropout(input, ...)` drop each element
with probability `p` and scale retained values by `1 / (1 - p)`. Evaluation and
`p=0` return the input. `p=1` returns zeros. A seed reproduces the sequence on the
same backend/build and execution order. CPU and CUDA use different generators.
Randomness advances only when the graph runs. Each functional call owns a sequence.
A module shares its sequence across forward calls on the same device. Repeated
Plans and CUDA replay draw fresh masks. Derivatives reuse the forward mask.
CPU uses `std::mt19937_64`. CUDA uses cuRAND's header-only Philox implementation
without a cuRAND runtime dependency.

## Module State And Mode

Modules start in training mode. `model.eval()` and `model.train()` recursively
change that mode. `model.training` reads it. `parameters()` returns trainable
tensors. `buffers()` returns non-trainable state, including BatchNorm's running
statistics. Python subclasses can register both explicitly. Mode is chosen when
building the tensor graph: rebuild a Plan after changing it. Running statistics
update only when their training graph executes, including gradient-only Plans.
Repeated calls to one BatchNorm layer within a graph update in construction
order. Do not execute Plans sharing mutable state or output storage concurrently.

`Tensor.rsqrt()` and `ops.rsqrt(input)` compute the reciprocal square root on
floating-point tensors. They support CPU/CUDA fusion and higher-order gradients.
Negative inputs produce NaN, and signed zero produces signed infinity.

```python
model = nn.Sequential([
    nn.Linear(16, 32, seed=1),
    nn.LayerNorm(32),
    nn.GELU(),
    nn.Linear(32, 5, seed=2),
])
logits = model(oc.Tensor(np.zeros((8, 16), dtype=np.float32)))
labels = oc.Tensor(np.arange(8, dtype=np.int32) % 5, dtype=oc.int32)
loss = nn.cross_entropy(logits, labels)
```

## Custom Layers

The C++ module headers are under `<ormi/nn/modules/...>`, activation functions
are in `<ormi/nn/activation.h>`, functional normalization is in
`<ormi/nn/normalization.h>`, and losses are under `<ormi/nn/losses/...>`.
All remain in namespace `ormi::nn` and link through
`ormi::core`. Custom C++ layers subclass `Module`, implement `forward()`, and
explicitly call the protected `register_parameter()` and `register_module()`
methods. Python layers subclass `nn.Module`, initialize its base, and implement
`forward()` using tensor operations:

```python
class Residual(nn.Module):
    def __init__(self, width, device="cpu"):
        super().__init__()
        self.layer = nn.Linear(width, width, device=device, seed=7)
        self.register_module(self.layer)

    def forward(self, x):
        return self.layer(x).sin() + x
```

Register each parameter and child explicitly after assigning its Python attribute.
Call `register_parameter(tensor)` for a custom realized floating-point leaf
parameter, and `register_module(layer)` for a child. A parent retains its
registered Python children, including their `forward()` methods. Each call
builds a graph. A prepared `Plan` runs its tensor operations directly.
Layers appended to a `Sequential` during `forward()` take effect on its next call.
`forward()` must return one tensor. Built-in module classes are final.
Subclass `Module` to define a new layer. Bindings, inheritance,
and type stubs are generated from `csrc/bindings/defs/nn.yaml`.

The [PINN example](../../ormi/examples/pinn.py) uses `Sequential`, `Linear`, `Sin`,
second input derivatives, and a reusable Plan. Each `Linear` has its own seed.
The loss accepts any `Module` with the required input and output shapes.
