# ormi

`ormi` exposes ORMI tensors, derivatives, and neural networks through Python.
It uses `ormi_core` objects: `ormi.Tensor is ormi_core.Tensor`. Python handles model
files. C++ handles tensor math and model state.

## Users

### Install

CPU:

```bash
python -m pip install "ormi==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

For CPU and CUDA support:

```bash
python -m pip install "ormi[cuda]==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Requirements

Use [CPython 3.10-3.13](https://www.python.org/downloads/) on x86-64 Windows
or Linux. See the [core requirements](../ormi-core/README.md#requirements)
for compiler, Linux glibc, GPU, and driver requirements.
CPU execution needs no C++ compiler. A supported C++20 compiler enables CPU
fusion. Without one, precompiled kernels keep graphs and derivatives lazy.
Wheels bundle OpenBLAS and SLEEF. The CUDA extra also bundles CUDA Runtime,
cuBLAS, and NVRTC. CUDA execution needs a supported GPU and NVIDIA driver.
Wheel users need no CMake, Ninja, or CUDA Toolkit.
`ormi` requires `ormi-core==0.1.0a8`. Its CUDA extra adds `ormi-cuda==0.1.0a8`.
[tqdm](https://tqdm.github.io/) is installed automatically and loaded when `fit` starts.

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
`from ormi import nn` and `from ormi.nn import Linear` use the same objects.
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

`fit` copies batches into reusable inputs, runs Plans, and reports loss and progress.
Matching batch shapes reuse a Plan. Short final batches are kept by default.
Set `drop_last=True` to skip them.
`history` maps reported epoch numbers to loss values.

Loss and update callbacks run only when building a graph for a new batch shape.
They build lazy tensor expressions. Call `model.eval()` before evaluation and
rebuild graphs after changing mode.

See the [training guide](docs/training.md) for callback rules, shuffling,
loss reporting, and execution limits.

## Differentiation

`grad(loss, input)` returns one tensor. `grad(loss, inputs)` returns a list.
The output must be scalar unless `seed` matches its shape, dtype, and device.
A seed weights the output derivatives (a vector-Jacobian product) without
building the full Jacobian.

For values shaped `(*P, outputs)` and points shaped `(*P, coordinates)`:

```python
u = model(points)
u_xx = om.partial(u, points, coordinate=0, order=2)
```

`partial` returns `(*P, 1)`. Entry `p` differentiates the selected output at
point `p` with respect to the selected input coordinate at `p`, including
dependencies through other points.
Specify `component` for multiple outputs. Selectors can be omitted only when
their dimension is one. Negative indices count from the end. `order` must be
a positive integer. Empty point dimensions work. Results use the input dtype.

For mixed partials, call `partial` successively with different coordinates.
Results stay lazy and support differentiation with respect to model parameters.
Each order builds one reverse pass per point, so graph size and execution cost
grow with point count. Index kernels use Int32 positions and reject larger
point axes or selected indices before execution. For fractional sampled-grid derivatives,
use `values.caputo(alpha, spacing, axis)`.

## Minibatches

For manual loops, `batches()` slices NumPy arrays along axis 0, keeping trailing
dimensions. Copy batches into matching inputs with `copy_from()`, then rerun their Plan.
The [minibatch guide](docs/training.md#minibatches) covers dtype/shape checks,
shuffling, and separate Plans for full and short batches.

## Model Checkpoints

```python
om.save(model, "model.npz")
om.load(model, "model.npz")
```

Create the model before loading. Checkpoints store named parameters and buffers.
Loading updates existing tensors. Rerun Plans to refresh outputs.
See [checkpoint rules](docs/training.md#model-checkpoints) for shared tensors,
file validation, and the additional state needed to resume training exactly.

## Examples

With `ormi` installed, run these from the root of a repository checkout:

```bash
python ormi/examples/pinn.py
python ormi/examples/minibatch.py
```

Examples show progress automatically. `--no-progress` selects plain-text logs.
The PINN example's `--report-every` controls loss sampling.

## Builders

### Development

Get a [source checkout and build prerequisites](../README.md#from-source).
From the repository root, set up the development environment:

```bash
uv sync
```

This creates `.venv`, installs development dependencies, and builds the CPU
core with CUDA, OpenBLAS, and SLEEF disabled. Run tests and type checks with:

```bash
uv run pytest
uv run pyright --project ormi/pyrightconfig.json --warnings
```

After editing C++ sources, rebuild the extension with
`uv sync --reinstall-package ormi-core`.
To install only the Python API with an already installed matching core, run
`python -m pip install --editable ./ormi` from the repository root.
The `ormi` package itself needs no C++ compiler.

Run `uv run pytest ormi/tests -q` to test only the Python API.
Pyright checks the PINN and minibatch examples and a strict typing fixture.
Run `uv run pytest ormi/tests/test_typing.py -q` to also check expected errors
for invalid calls and module subclasses.
Editor signatures come from the core's YAML-generated stubs. The submodule
files re-export those types for `ormi.nn`, `ormi.ops`, `ormi.optim`, and `ormi.jit`.

### Release Builds

The release workflow builds the source archive and wheel, then tests wheel
installs on Windows and Linux with Python 3.10-3.13 before publishing.
Tags use `ormi-core-vVERSION`.
