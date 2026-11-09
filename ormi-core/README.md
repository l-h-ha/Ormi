# ormi-core

`ormi-core` builds and runs tensor graphs and computes their derivatives.
It provides the C++ library and the `ormi_core` Python module on x86-64
Windows and Linux. For Python training, see the [ormi guide](../ormi/README.md).

## Users

### Install

CPU:

```bash
python -m pip install "ormi-core==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

CPU and CUDA:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a7" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

### Usage

```python
import ormi_core as oc

x = oc.Tensor([1.0, 2.0, 3.0])
y = (x * x + 1.0).sum()
dx = oc.grad(y, x)
oc.Plan([y, dx]).run()
print(y.item(), dx.numpy())
```

`Plan.run()` evaluates the graph. Reading an unrealized tensor through
`item()` or `numpy()` also evaluates it.

### Requirements

| Use | Requirement |
| --- | --- |
| Python | [CPython 3.10-3.13](https://www.python.org/downloads/) and NumPy 1.24 or newer. |
| Windows | 64-bit Windows. |
| Linux | x86-64 Linux with glibc 2.28 or newer. |
| CPU | Release wheels bundle OpenBLAS 0.3.33 and SLEEF 3.9.0. No separate installation is needed. |
| CPU fusion | Optional [C++20 compiler](#cpu-fusion-compiler). Without one, execution uses precompiled CPU kernels. |
| CUDA | The CUDA extra, a supported NVIDIA GPU, and a compatible [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/). No CUDA Toolkit. |

Using these wheels does not require a C++ compiler, CMake, or Ninja.
A compatible compiler enables CPU fusion. Precompiled execution preserves
lazy graphs, reusable Plans, training, and higher-order derivatives, with
more separate kernel calls and intermediate buffers than fusion.

The release configuration uses CUDA 12.4. The minimum supported driver is
`550.54.14` on Linux and `551.61` on Windows. The wheels target compute
capabilities 5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, and 9.0, with PTX
for newer 9.x devices. Float16 fusion requires compute capability 5.3 or newer.

CUDA loads only when a CUDA device is requested. Importing `ormi_core` and
using CPU tensors does not require CUDA.

Compiler installers and instructions:

- Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  (Visual Studio 2022 is tested),
  [LLVM/Clang](https://releases.llvm.org/), or
  [MSYS2 UCRT64 GCC](https://www.msys2.org/).
- Linux: [GCC](https://gcc.gnu.org/install/) or
  [LLVM/Clang](https://releases.llvm.org/). Ubuntu users can follow the
  [Ubuntu GCC setup guide](https://ubuntu.com/developers/docs/howto/gcc-setup/).

## Builders

| Feature | Requirement |
| --- | --- |
| Core | [CMake 3.18+](https://cmake.org/download/) and a C++20 compiler. |
| Presets | CMake 3.21+. Presets such as `cpu-debug` and `cuda-debug` use [Ninja](https://ninja-build.org/). The MSVC presets use the Visual Studio generator. |
| Linux toolchains tested in CI | GCC 13 and Clang 18 on Ubuntu 24.04. |
| Windows toolchains tested in CI | Visual Studio 2022 MSVC, clang-cl, and MSYS2 UCRT64 GCC. |
| Python build | CPython 3.10-3.13. Build isolation installs scikit-build-core 1.0+, pybind11 3.0+, and PyYAML 6.0+. |
| Tests and benchmarks | The first configure needs network access for CMake to fetch GoogleTest and Google Benchmark. Tests also need a matching C compiler. |
| Sanitizers | Windows x64: Visual Studio 2022 MSVC with [C++ AddressSanitizer](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170). Linux: Clang with [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html). |
| OpenBLAS | Optional. Requires exactly [OpenBLAS 0.3.33](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.33) with a discoverable CMake package. |
| SLEEF | Optional. Requires exactly [SLEEF 3.9.0](https://github.com/shibatch/sleef/releases/tag/3.9.0) with a discoverable CMake package. |
| CUDA | Optional. Requires [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) with C++20-capable NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, NVRTC, and cuRAND headers. Release wheels use Toolkit 12.4. |

Only Windows and Linux have platform implementations.

From the repository root, enter the core directory:

```bash
cd ormi-core
```

Run the commands below from that directory unless another location is stated.
On Windows, use a shell configured for your compiler. MSVC and CUDA builds
use the Visual Studio 2022 developer shell.

### CPU

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Compiler-specific presets are `cpu-gcc-debug`, `cpu-clang-debug`,
`cpu-msvc-debug`, and `cpu-clang-cl-debug`. Release benchmarks use
`cpu-bench`.

Run CPU tests with AddressSanitizer on Windows x64 from a Visual Studio 2022
developer shell:

```powershell
cmake --preset cpu-msvc-sanitize
cmake --build --preset cpu-msvc-sanitize
ctest --preset cpu-msvc-sanitize
```

On Linux, use Clang AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake --preset cpu-clang-sanitize
cmake --build --preset cpu-clang-sanitize
ctest --preset cpu-clang-sanitize
```

### OpenBLAS

```bash
cmake -S . -B build-openblas -G Ninja -DORMI_ENABLE_OPENBLAS=ON -DCMAKE_PREFIX_PATH=/path/to/openblas
cmake --build build-openblas --parallel
ctest --test-dir build-openblas --output-on-failure
```

On Windows, replace `CMAKE_PREFIX_PATH` with:

```text
-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

The repository's `vcpkg.json` pins the Windows dependency used by release
builds. Follow Microsoft's [vcpkg setup guide](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)
to install vcpkg.

### SLEEF

Enable vectorized Float32 `exp`, `log`, `sin`, and `cos` with:

```bash
cmake -S . -B build-sleef -G Ninja -DORMI_ENABLE_SLEEF=ON -DCMAKE_PREFIX_PATH=/path/to/sleef
cmake --build build-sleef --parallel
ctest --test-dir build-sleef --output-on-failure
```

On Windows, the vcpkg toolchain in the preceding example supplies SLEEF as
well as OpenBLAS.

### CUDA

The CUDA preset enables `ORMI_ENABLE_CUDA`. The CPU presets leave it disabled.
For a custom configuration, set `-DORMI_ENABLE_CUDA=ON` explicitly.

```bash
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

On Windows, run this from a Visual Studio 2022 developer shell. Release
benchmarks use `cuda-bench`.

### Python

Build and install the CPU Python package from `ormi-core/`:

```bash
python -m pip install .
```

That source build has OpenBLAS, SLEEF, and CUDA disabled unless explicitly
configured.
Build the separately installable CUDA wheel from the repository root with:

```bash
python -m pip wheel ./ormi-cuda
```

### C++

For C++, use CMake `add_subdirectory` and link `ormi::core`. When CUDA is
enabled, C++ applications that call the CUDA backend directly may also link
`ormi::cuda`. A `find_package(ormi)` package is not installed yet.

## Runtime Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `ORMI_CPU_THREADS` | Detected physical cores | Threads used by large contiguous CPU JIT maps. Set to `1` for serial execution. |
| `ORMI_CPU_JIT_CXX` | Unset | Compiler executable name or path. |
| `ORMI_CPU_JIT_CXX_ID` | Auto | Compiler syntax: `gcc`, `clang`, `clang-cl`, or `msvc`. |
| `ORMI_CPU_JIT_TARGET` | `auto` | CPU instruction set: `auto`, `baseline`, or `avx2`. |
| `ORMI_CPU_JIT_TIMEOUT_MS` | `120000` | Compile timeout in milliseconds. Set to `0` to disable it. |
| `ORMI_CPU_JIT_CACHE_DIR` | User cache | Persistent fused-kernel cache directory. |
| `ORMI_CPU_JIT_CACHE_MAX_BYTES` | `268435456` | Cache budget in bytes. Set to `0` to disable eviction. |
| `ORMI_CUDA_JIT_CACHE_DIR` | User cache | Persistent CUDA source and PTX cache directory. |
| `ORMI_CUDA_JIT_CACHE_MAX_BYTES` | `268435456` | CUDA cache budget in bytes. Set to `0` to disable eviction. |

Bindings and `.pyi` files are generated from `csrc/bindings/defs/*.yaml`.

## CPU Fusion Compiler

Compiler lookup happens when CPU execution first considers a fusible graph.
Graph construction and differentiation do not look for a compiler:

1. `ORMI_CPU_JIT_CXX`, when set, supplies the executable name or path.
2. Windows tries `cl.exe`, `clang-cl.exe`, `g++.exe`, and `clang++.exe`, then
   searches for Visual Studio.
3. Linux tries `g++`, `clang++`, and `c++`.
4. If no compatible compiler is found, ORMI uses an existing cached kernel
   when available, or schedules the graph's precompiled operations.

Automatically selected compilers must pass a C++20 syntax and header check.
Setting `ORMI_CPU_JIT_CXX` skips that check. A missing executable selects
precompiled execution. ORMI does not search for another compiler.
Invalid compiler settings, an unidentified executable, and failures from a
selected compiler still raise `jitError`. Set compiler variables before
first execution or Plan construction, when ORMI caches the compiler decision.

With `ORMI_CPU_JIT_CXX` set, `ORMI_CPU_JIT_CXX_ID` specifies its command-line
syntax. Without `ORMI_CPU_JIT_CXX`, it restricts automatic selection to that
compiler family. Values are case-insensitive: `gcc`, `clang`, `clang-cl`, and
`msvc`. `cl` is an alias for `msvc`.

`ORMI_CPU_JIT_TARGET` selects the generated CPU instruction set. Its default,
`auto`, uses AVX2 when the CPU and operating system support it and otherwise
uses the baseline instruction set. `baseline` and `avx2` are explicit
case-insensitive overrides. Requesting unsupported AVX2 raises `jitError`
when CPU execution first considers fusion.

On AVX2 CPUs with F16C, supported fused kernels process several Float16 values
at once. F16C is detected automatically. CPUs without it process Float16 values
one at a time.

Release wheels use SLEEF 3.9.0 for eight-lane Float32 `exp`, `log`, `sin`,
and `cos` in supported AVX2 fused kernels. ORMI passes these functions to
generated kernels, so the user's JIT compiler does not
need SLEEF headers or libraries. Non-AVX2 systems use scalar standard-library
math.

## Neural Networks

`ormi_core.nn` provides layers, activations, losses, and normalization.
See the [NN reference](docs/nn.md) for shapes, initialization, gradients,
module state, and custom layers. The [Python guide](../ormi/README.md)
covers training and model checkpoints through `ormi`.

## License

MIT. Bundled dependencies retain their own licenses.
