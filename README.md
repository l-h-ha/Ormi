# ORMI

ORMI is an alpha-stage lazy tensor and automatic-differentiation library for
Python and C++. It currently supports x86-64 Windows and Linux.

The Python package is `ormi_core`.

## Users

Install a supported [CPython 3.10-3.13](https://www.python.org/downloads/),
then install ORMI.

Install CPU support:

```bash
python -m pip install "ormi-core==0.1.0a6" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

Install CPU and CUDA 12.4 support:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a6" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

The wheels include OpenBLAS, SLEEF vector math, and, for the CUDA extra, ORMI's
CUDA runtime libraries. Users do not install OpenBLAS, SLEEF, or the CUDA
Toolkit. CUDA users only install a compatible
[NVIDIA driver](https://www.nvidia.com/Download/index.aspx/).
CPU fusion needs a C++20 compiler when an uncached fused kernel is first used.
CUDA fusion stores generated PTX in the user's cache, so NVRTC compilation is
normally paid once per kernel and GPU target.

Read [ormi-core/README.md](ormi-core/README.md) for the exact Python, compiler,
platform, GPU, and NVIDIA driver requirements.

## Builders

Install [CMake 3.18 or newer](https://cmake.org/download/),
[Ninja](https://ninja-build.org/), and a C++20 toolchain:

- Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  (Visual Studio 2022 is tested),
  [LLVM/Clang](https://releases.llvm.org/), or
  [MSYS2 UCRT64 GCC](https://www.msys2.org/).
- Linux: [GCC](https://gcc.gnu.org/install/) or
  [LLVM/Clang](https://releases.llvm.org/).

Then run:

```bash
cd ormi-core
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

CUDA additionally requires the
[CUDA 12.4 Toolkit](https://developer.nvidia.com/cuda-toolkit-archive). It is
enabled only by the CUDA preset:

```bash
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

For Python workspace development, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
and run from the repository root:

```bash
uv sync
uv run pytest
```

`uv sync` builds `ormi-core` with CUDA and OpenBLAS disabled. Detailed build
requirements and optional build commands are in
[ormi-core/README.md](ormi-core/README.md).

## Layout

- `ormi-core`: tensor, autograd, CPU runtime, Python bindings, and CUDA source.
- `ormi-cuda`: separately installable CUDA plugin package.

## License

MIT.
