# ormi

`ormi` is the Python API for ORMI tensors, derivatives, and neural networks.
It uses `ormi_core` objects directly: `ormi.Tensor is ormi_core.Tensor`.
Python handles model files. C++ handles tensor math and model state.

## Users

### Install

CPU:

```bash
python -m pip install "ormi==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

For CPU and CUDA support:

```bash
python -m pip install "ormi[cuda]==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Requirements

Use [CPython 3.10-3.13](https://www.python.org/downloads/) on x86-64 Windows
or Linux. See the [core requirements](../ormi-core/README.md#requirements)
for compiler, Linux glibc, GPU, and driver requirements.
CPU use does not require a C++ compiler. A supported C++20 compiler enables
CPU fusion. Without one, ORMI uses precompiled kernels and keeps graphs and
derivatives lazy.
Wheels bundle OpenBLAS and SLEEF. The CUDA extra also bundles CUDA Runtime,
cuBLAS, and NVRTC. CUDA execution needs a supported GPU and NVIDIA driver.
Users do not need CMake, Ninja, or the CUDA Toolkit to use these wheels.
`ormi` requires `ormi-core==0.1.0a7`. Its CUDA extra adds `ormi-cuda==0.1.0a7`.
Installation also includes [tqdm](https://tqdm.github.io/), which `fit` loads
when training starts.

## Usage

```python
import ormi as om
from ormi import nn

model = nn.Sequential([nn.Linear(3, 8, seed=7), nn.Tanh(), nn.Linear(8, 1, seed=8)])
x = om.tensor([[0.1, 0.2, 0.3]])
y = model(x)
dx = om.grad(y.sum(), x)
om.Plan([y, dx]).run()
print(y.numpy())
```

Model calls and derivatives build graphs. `Plan.run()` evaluates them.
`nn`, `ops`, `optim`, and `jit` refer directly to the core modules.
Both `from ormi import nn` and `from ormi.nn import Linear` use the same objects.
The [NN reference](../ormi-core/docs/nn.md) covers layers, activations, losses,
normalization, and custom modules.

## Training

```python
import numpy as np
import ormi as om
from ormi import nn

values = np.random.default_rng(7).normal(size=(100, 3)).astype(np.float32)
labels = np.argmax(values, axis=-1).astype(np.int32)
model = nn.Sequential([nn.Linear(3, 8, seed=7), nn.Tanh(), nn.Linear(8, 3, seed=8)])

def loss(inputs, targets):
    return nn.cross_entropy(model(inputs), targets)

history = om.fit(
    model, values, labels, loss=loss,
    update=lambda parameter, gradient: om.sgd(parameter, gradient, 0.01),
    epochs=30, batch_size=32, seed=7,
)
```

`fit` copies batches into reusable input tensors, runs Plans, and reports loss
with a built-in progress display. It keeps the short final batch by default.
Set `drop_last=True` to skip it. Batches with the same shape reuse a Plan.
`history` maps reported epoch numbers to loss values.

Loss and update callbacks build lazy tensor expressions. They run when building
the graph for a new batch shape. Call `model.eval()` before evaluation and
build a new graph after changing the mode.

See the [training guide](docs/training.md) for callback rules, shuffling,
loss reporting, and execution limits.

## Differentiation

`grad(loss, input)` returns one tensor. `grad(loss, inputs)` returns a list.
The output must be scalar unless an explicit `seed` with matching shape,
dtype, and device is supplied. A seed gives a weighted sum of output derivatives
(a vector-Jacobian product), without building the full Jacobian matrix.

For values shaped `(*P, outputs)` and points shaped `(*P, coordinates)`:

```python
u = model(points)
u_xx = om.partial(u, points, coordinate=0, order=2)
```

`partial` returns `(*P, 1)`. Entry `p` is the derivative of the selected output
at point `p` with respect to the selected coordinate at that same point. It
includes dependencies through other points.
Specify `component` for multiple outputs. A selector can be omitted only when
its dimension is one. Negative indices count from the end. `order` is a positive
integer. Empty point dimensions are supported, and the result uses the input dtype.

Mixed partials use successive calls with different coordinates. All results
are lazy tensors and can be differentiated with respect to model parameters.
The exact implementation builds one reverse pass per point per order, so its
graph and execution cost grow with the number of points. Current index kernels
use Int32 positions. An individual point axis or selected index beyond that
range is rejected before execution. Fractional sampled-grid differentiation
remains a separate operation: `values.caputo(alpha, spacing, axis)`.

## Minibatches

For a manual training loop, `batches()` slices NumPy arrays along axis 0 and
keeps every trailing dimension. Copy compatible batches into existing input
tensors with `copy_from()` before running their Plan again.
The [minibatch guide](docs/training.md#minibatches) covers dtype/shape checks,
shuffling, and separate Plans for full and short batches.

## Model Checkpoints

```python
om.save(model, "model.npz")
om.load(model, "model.npz")
plan.run()
```

Create the model before loading. Checkpoints store named parameters and buffers.
Loading writes into existing tensors. Run the Plan again to refresh outputs.
See [checkpoint rules](docs/training.md#model-checkpoints) for shared tensors,
file validation, and the additional state needed to resume training exactly.

## Builders

### Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the
[core build prerequisites](../ormi-core/README.md#builders).
From the repository root:

```bash
uv sync
uv run pytest
uv run pyright --project ormi/pyrightconfig.json --warnings
```

`uv sync` builds the CPU core with CUDA, OpenBLAS, and SLEEF disabled.
To use an already installed matching core, run `python -m pip install ./ormi`
from the repository root. Building the `ormi` wheel itself needs no C++ compiler.

Run `uv run pytest ormi/tests -q` to test only the Python API.
Pyright checks the PINN and minibatch examples and a strict typing fixture.
Run `uv run pytest ormi/tests/test_typing.py -q` to also check expected errors
for invalid calls and module subclasses.
Editor signatures come from the core's YAML-generated stubs. The submodule
files re-export those types for `ormi.nn`, `ormi.ops`, `ormi.optim`, and `ormi.jit`.

### Examples

From the repository root after `uv sync`:

```bash
uv run python ormi/examples/pinn.py
uv run python ormi/examples/minibatch.py
```

Examples show progress automatically. `--no-progress` selects plain-text logs.
The PINN example's `--report-every` controls loss sampling.

### Release Builds

The release workflow builds this package's source archive and Python wheel,
then tests installed wheels on Windows and Linux with Python 3.10-3.13.
Publishing waits for those tests. Tags keep the form `ormi-core-vVERSION`.
