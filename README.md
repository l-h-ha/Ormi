# ORMI

ORMI is an alpha tensor library for Python and C++. It builds operations as a
graph, runs them when needed, and computes derivatives automatically.
It supports x86-64 Windows and Linux.

`ormi` exposes `ormi_core` objects as the Python API. `ormi-core` contains the
C++ runtime and bindings. `ormi-cuda` adds CUDA support.

## Users

Install [CPython 3.10-3.13](https://www.python.org/downloads/) on x86-64
Windows or Linux.

### Install

CPU:

```bash
python -m pip install "ormi==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

CPU and CUDA 12.8:

```bash
python -m pip install "ormi[cuda]==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Requirements

CPU wheels bundle OpenBLAS and SLEEF. Without a compatible compiler, ORMI uses
precompiled CPU kernels. A supported C++20 compiler enables CPU fusion. See the
[exact requirements](ormi-core/README.md#requirements).

The CUDA extra bundles CUDA Runtime, cuBLAS, and NVRTC. CUDA execution requires
a supported NVIDIA GPU and [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/).
Wheel users need no CMake, Ninja, or CUDA Toolkit.

## Example

```python
import ormi as om

x = om.tensor([1.0, 2.0, 3.0])
y = (x * x).sum()
dx = om.grad(y, x)
om.Plan([y, dx]).run()
print(y.item(), dx.numpy())
```

Operations and derivatives stay lazy until their graph runs. The
[Python guide](ormi/README.md) covers training, batching, model checkpoints,
and higher-order derivatives. See [examples](ormi/README.md#examples) for
PINN and minibatch training scripts.

## Builders

### From Source

Install [Git](https://git-scm.com/downloads),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a supported
[C++20 compiler](ormi-core/README.md#compilers).
On Windows with MSVC, use an x64 Visual Studio 2022 developer shell.

```bash
git clone https://github.com/l-h-ha/Ormi.git
cd Ormi
uv sync
```

For an existing checkout, run `uv sync` from its root.
It creates `.venv`, installs development dependencies, and builds the CPU
core with CUDA, OpenBLAS, and SLEEF disabled. Use `uv run` for commands in
this environment. Python builds install CMake and Ninja when needed.

For C++ builds, compiler-specific presets, sanitizers, and optional OpenBLAS,
SLEEF, or CUDA support, follow the [core build instructions](ormi-core/README.md#builders).
For Python workspace tests, follow [Python development](ormi/README.md#development).

## Packages

- [ormi](ormi/README.md): Python usage, training, and examples.
- [ormi-core](ormi-core/README.md): core usage, prerequisites, C++ builds, and runtime settings.
- [ormi-cuda](ormi-cuda/README.md): CUDA plugin installation and builds.

## License

MIT.
