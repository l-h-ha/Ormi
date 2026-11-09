# ORMI

ORMI is an alpha tensor library for Python and C++. It builds operations as a
graph, runs them when needed, and computes derivatives automatically.
It supports x86-64 Windows and Linux.

`ormi` provides the Python API and uses `ormi_core` objects directly.
`ormi-core` contains the C++ runtime and Python bindings. `ormi-cuda` adds
the optional CUDA backend.

## Users

Install [CPython 3.10-3.13](https://www.python.org/downloads/) on x86-64
Windows or Linux.

### Install

CPU:

```bash
python -m pip install "ormi==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

CPU and CUDA 12.4:

```bash
python -m pip install "ormi[cuda]==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Requirements

CPU wheels bundle OpenBLAS and SLEEF. A C++ compiler is optional: ORMI uses
precompiled CPU kernels when no compatible compiler is available. Installing
a supported C++20 compiler enables CPU fusion. See the
[exact requirements](ormi-core/README.md#requirements).

The CUDA extra bundles CUDA
Runtime, cuBLAS, and NVRTC. CUDA execution needs a supported NVIDIA GPU and an installed
[NVIDIA driver](https://www.nvidia.com/Download/index.aspx/).

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
and higher-order derivatives.

## Builders

### From Source

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a
supported [C++20 compiler](ormi-core/README.md#cpu-fusion-compiler).
From the repository root, run:

```bash
uv sync
uv run python ormi/examples/pinn.py
```

This builds the CPU core with CUDA, OpenBLAS, and SLEEF disabled.
See the [build requirements](ormi-core/README.md#builders) for CMake, Ninja,
and optional dependencies.

For C++ builds, compiler-specific presets, sanitizers, and optional OpenBLAS,
SLEEF, or CUDA support, follow the [core build instructions](ormi-core/README.md#builders).
For Python workspace tests, follow [Python development](ormi/README.md#development).

## Packages

- [ormi](ormi/README.md): Python usage, training, and examples.
- [ormi-core](ormi-core/README.md): core usage, prerequisites, C++ builds, and runtime settings.
- [ormi-cuda](ormi-cuda/README.md): CUDA plugin installation and builds.

## License

MIT.
