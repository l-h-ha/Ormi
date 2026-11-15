# ormi-core

`ormi-core` builds and runs tensor and derivative graphs. It provides the C++
library and `ormi_core` Python module on x86-64 Windows and Linux.
For Python training, see the [ormi guide](../ormi/README.md).

## Users

### Install

CPU:

```bash
python -m pip install "ormi-core==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

CPU and CUDA:

```bash
python -m pip install "ormi-core[cuda]==0.1.0a8" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
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

`Plan.run()` runs the graph. For unrealized tensors, `item()` and `numpy()`
also run their graphs.

Index tensors with integers, slices, or `...`, for example `x[1:]` or
`matrix[:, ::-1]`. Positive-step slices share storage. Reversed slices make a
lazy copy. Array indices, boolean indices, and `None` are unsupported.
`oc.concat([a, b], axis=0)` joins tensors with matching rank, dtype, and device.
Dimensions outside the selected axis must match. Both operations preserve
higher-order differentiation.

### Requirements

| Use | Requirement |
| --- | --- |
| Python | [CPython 3.10-3.13](https://www.python.org/downloads/) and NumPy 1.24 or newer. |
| Windows | x86-64 Windows. |
| Linux | x86-64 Linux with glibc 2.28 or newer. |
| CPU | Release wheels bundle OpenBLAS 0.3.33 and SLEEF 3.9.0. No separate installation is needed. |
| CPU fusion | Optional [C++20 compiler](#cpu-fusion-compiler). Without one, execution uses precompiled CPU kernels. |
| CUDA | The CUDA extra, a supported NVIDIA GPU, and a compatible [NVIDIA driver](https://www.nvidia.com/Download/index.aspx/). No CUDA Toolkit. |

Wheel users need no C++ compiler, CMake, or Ninja. Precompiled kernels support
lazy graphs, Plan reuse, training, and higher-order derivatives. CPU fusion
reduces kernel calls and intermediate buffers.

Version 0.1.0a8 wheels use CUDA 12.8. CUDA JIT requires driver `570.124.06` or
newer on Linux or `572.61` or newer on Windows. The wheels target compute
capabilities 5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, and 10.0,
with PTX for newer devices. Fusion selects the highest NVRTC-supported target
no greater than the GPU's compute capability. Float16 fusion requires compute
capability 5.3 or newer.

CUDA loads on first CUDA use. CPU tensors and `ormi_core` imports need no CUDA.

### Compilers

For source builds or optional CPU fusion, install one of these compilers:

- Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  (Visual Studio 2022 is tested),
  [LLVM/Clang](https://releases.llvm.org/), or
  [MSYS2 UCRT64 GCC](https://www.msys2.org/).
- Linux: [GCC](https://gcc.gnu.org/install/) or
  [LLVM/Clang](https://releases.llvm.org/). Ubuntu users can follow the
  [Ubuntu GCC setup guide](https://ubuntu.com/developers/docs/howto/gcc-setup/).

## Builders

Start with a [source checkout](../README.md#from-source).

| Feature | Requirement |
| --- | --- |
| C++ library | [CMake 3.18+](https://cmake.org/download/) and a C++20 compiler. |
| Presets | CMake 3.21+. Presets such as `cpu-debug` and `cuda-debug` use [Ninja](https://ninja-build.org/). The MSVC presets use the Visual Studio generator. |
| Linux toolchains tested in CI | GCC 13 and Clang 18 on Ubuntu 24.04. |
| Windows toolchains tested in CI | Visual Studio 2022 MSVC, clang-cl, and MSYS2 UCRT64 GCC. |
| Python build | CPython 3.10-3.13 with development headers. Build isolation installs scikit-build-core 1.0+, pybind11 3.0+, and PyYAML 6.0+. |
| Tests and benchmarks | The first configure needs network access for CMake to fetch GoogleTest and Google Benchmark. Tests also need a matching C compiler. |
| Sanitizers | Windows x64: Visual Studio 2022 MSVC with [C++ AddressSanitizer](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170). Linux: Clang with [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) and [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html). |
| OpenBLAS | Optional. Requires exactly [OpenBLAS 0.3.33](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.33) with a discoverable CMake package. |
| SLEEF | Optional. Requires exactly [SLEEF 3.9.0](https://github.com/shibatch/sleef/releases/tag/3.9.0) with a discoverable CMake package. |
| CUDA | Optional. Requires [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) with C++20-capable NVCC, CUDA Runtime, CUDA Driver import library, cuBLAS, NVRTC, and cuRAND headers. The current wheel-build configuration uses Toolkit 12.8. |

Only Windows and Linux have platform implementations.
Python builds use [scikit-build-core](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html)
to install build tools as needed. For the CMake commands below, install CMake
and the selected build tool yourself.

Wheels built with the current CUDA 12.8 configuration require driver
`570.124.06` or newer on Linux or `572.61` or newer on Windows for CUDA JIT.
These are the [CUDA 12.8 Update 1 driver versions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).
They retain the existing GPU targets and add compute capability 10.0.
Fusion selects the highest generic PTX target supported by both NVRTC and the
GPU. A newer GPU can use an older PTX target through the driver JIT.

From the repository root, enter the core directory:

```bash
cd ormi-core
```

Run the remaining commands there unless stated otherwise.
On Windows, use a shell configured for your compiler. MSVC and CUDA builds
use an x64 Visual Studio 2022 developer shell.

### CPU

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
```

Run the C++ tests:

```bash
ctest --preset cpu-debug
```

This preset uses Ninja and leaves CUDA, OpenBLAS, and SLEEF disabled.
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

Replace `/path/to/openblas` with the directory where OpenBLAS is installed.
On Windows, replace the `-DCMAKE_PREFIX_PATH=/path/to/openblas` argument with:

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

Replace `/path/to/sleef` with the directory where SLEEF is installed.
On Windows, replace that prefix argument with the vcpkg toolchain argument
above. It supplies SLEEF as well as OpenBLAS.

### CUDA

The CUDA preset enables `ORMI_ENABLE_CUDA`. The CPU presets leave it disabled.
For a custom configuration, set `-DORMI_ENABLE_CUDA=ON` explicitly.
Running CUDA tests requires a supported NVIDIA GPU and driver.

```bash
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

On Windows, run this from an x64 Visual Studio 2022 developer shell. Release
benchmarks use `cuda-bench`.

### Python

Build and install the CPU Python package from `ormi-core/`:

```bash
python -m pip install .
```

That source build has OpenBLAS, SLEEF, and CUDA disabled unless explicitly
configured. It installs `ormi_core`. For the Python training API, use the
[workspace build](../README.md#from-source).
Build the optional CUDA plugin with the
[CUDA wheel instructions](../ormi-cuda/README.md#wheel).

Bindings and `.pyi` files are generated from `csrc/bindings/defs/*.yaml`.

### C++

Use CMake `add_subdirectory` and link `ormi::core`. Applications calling the
CUDA backend directly can also link `ormi::cuda` when enabled.
There is no `find_package(ormi)` package yet.

## Runtime Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `ORMI_CPU_THREADS` | Detected physical cores | CPU worker count. Set to `1` for serial execution. |
| `ORMI_CPU_JIT_CXX` | Unset | Compiler executable name or path. |
| `ORMI_CPU_JIT_CXX_ID` | Auto | Compiler syntax: `gcc`, `clang`, `clang-cl`, or `msvc`. |
| `ORMI_CPU_JIT_TARGET` | `auto` | CPU instruction set: `auto`, `baseline`, or `avx2`. |
| `ORMI_CPU_JIT_TIMEOUT_MS` | `120000` | Compile timeout in milliseconds. Set to `0` to disable it. |
| `ORMI_CPU_JIT_CACHE_DIR` | User cache | Persistent fused-kernel cache directory. |
| `ORMI_CPU_JIT_CACHE_MAX_BYTES` | `268435456` | Cache budget in bytes. Set to `0` to disable eviction. |
| `ORMI_CUDA_JIT_CACHE_DIR` | User cache | Persistent CUDA source and PTX cache directory. |
| `ORMI_CUDA_JIT_CACHE_MAX_BYTES` | `268435456` | CUDA cache budget in bytes. Set to `0` to disable eviction. |

## CPU Fusion Compiler

Compiler lookup starts when CPU execution first considers fusion.
Tensor graph construction and differentiation do not search for compilers:

1. `ORMI_CPU_JIT_CXX`, when set, supplies the executable name or path.
2. Windows tries `cl.exe`, `clang-cl.exe`, `g++.exe`, and `clang++.exe`, then
   searches for Visual Studio.
3. Linux tries `g++`, `clang++`, and `c++`.
4. If no compatible compiler is found, ORMI uses an existing cached kernel
   when available, or schedules the graph's precompiled operations.

Auto-selected compilers must pass a C++20 syntax and header check.
Setting `ORMI_CPU_JIT_CXX` skips it. A missing executable selects precompiled
kernels without trying another compiler. Invalid settings, unidentified executables,
and compiler failures raise `jitError`. Set compiler variables before first
execution or Plan construction. ORMI caches the choice.

With `ORMI_CPU_JIT_CXX` set, `ORMI_CPU_JIT_CXX_ID` selects its command-line
syntax. Otherwise it restricts discovery to that compiler family.
Values are case-insensitive: `gcc`, `clang`, `clang-cl`, and
`msvc`. `cl` is an alias for `msvc`.

`ORMI_CPU_JIT_TARGET` selects CPU instructions. `auto` uses AVX2 when supported
by the CPU and operating system, otherwise baseline instructions. Overrides
`baseline` and `avx2` are case-insensitive. Unsupported AVX2 raises `jitError` when CPU execution
first considers fusion.

On AVX2 CPUs with F16C, supported fused kernels process several Float16 values
at once. F16C is detected automatically. CPUs without it process Float16 values
one at a time.

Release wheels use SLEEF 3.9.0 for eight-lane Float32 `exp`, `log`, `sin`,
and `cos` in supported AVX2 fused kernels. ORMI passes these functions to
generated kernels. The JIT compiler needs no SLEEF headers or libraries.
Non-AVX2 systems use scalar standard-library math.

## Neural Networks

`ormi_core.nn` provides layers, activations, losses, and normalization.
See the [NN reference](docs/nn.md) for shapes, initialization, gradients,
module state, and custom layers. The [Python guide](../ormi/README.md)
covers training and model checkpoints through `ormi`.

## License

MIT. Bundled dependencies retain their own licenses.
