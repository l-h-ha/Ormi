# ORMI

ORMI is an early-stage Python tensor and autograd project. It is designed around lazy computation: tensor operations build computation graphs
that can be executed, optimized, and accelerated by the backend.

This repository is a `uv` workspace. For now, the only implemented package is
`ormi-core`: a Python backend library powered by a C++/CUDA backend.
The top-level `ormi` package exists in the workspace, but it has not started
development yet.

## Packages

- `ormi-core`:
C++/CUDA tensor and autograd engine with Python bindings.
- `ormi`: Placeholder package for the future user-facing ORMI API.

For package-specific documentation, see:

- [ormi-core/README.md](ormi-core/README.md)

## Set Up The Workspace

This repository is a real `uv` workspace. The root project depends on the
workspace packages through `[tool.uv.sources]`, so `uv sync` installs `ormi`
and `ormi-core` from the local checkout.

Because `ormi-core` contains a compiled extension, syncing the full workspace may
compile it. The default source build is CPU-only and requires:

- CMake `>=3.18`.
- A C++20 compiler.
- CPython `>=3.10`.

Then install the workspace dependencies from the repository root:

```bash
uv sync
```

This installs the workspace packages in editable mode and prepares the local
development environment.

OpenBLAS and CUDA are explicit optional source-build features. On Windows,
enable exact OpenBLAS `0.3.33` through the pinned vcpkg manifest with:

```powershell
$env:CMAKE_GENERATOR = "Ninja"
$env:SKBUILD_CMAKE_ARGS = "-DORMI_ENABLE_OPENBLAS=ON;-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
uv sync
```

Make sure Ninja is installed and available on `PATH` if you set
`CMAKE_GENERATOR` to `Ninja`.

## Run Tests

After the workspace is synced, run the Python test suite from the repository
root:

```bash
uv run pytest
```

Most tests live under `ormi-core`. See the `ormi-core` README for the
full CMake, C++ test, CUDA, OpenBLAS, and platform-specific build notes.

## Build Packages

To build the implemented backend package:

```bash
uv build --package ormi-core
```

This command uses the same build prerequisites as `uv sync`.

To build all workspace packages:

```bash
uv build --all-packages
```

Right now, `ormi-core` is the package that matters for contributors. The
`ormi` package is present so the workspace shape is ready in the future.

## Development

If you wish to contribute, start with the checked CPU preset:

```bash
cd ormi-core
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

CUDA is never enabled just because a GPU or toolkit is present. Configure it
explicitly with `-DORMI_ENABLE_CUDA=ON` or use the `cuda-debug` preset.

For the full build guide, including CUDA and OpenBLAS details, read:

- [ormi-core/README.md](ormi-core/README.md)

## Repository Layout

```text
ormi-workspace/
  ormi-core/      Implemented Python backend library
  ormi/           Future user-facing package; not developed yet
  pyproject.toml  uv workspace configuration
  uv.lock         Locked workspace dependencies
```

## License

MIT
