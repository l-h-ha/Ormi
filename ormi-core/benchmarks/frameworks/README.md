# Framework benchmarks

This corpus compares ORMI, PyTorch, and TensorFlow on the same deterministic
NumPy inputs. It is evidence collection, not a single headline benchmark.

## Workloads

| Workload | Timed work |
| --- | --- |
| `pointwise` | Broadcast multiply, add, subtract, and multiply |
| `reduction` | Broadcast pointwise work followed by a two-axis mean |
| `dense` | Dense mean-squared error and gradients for weight and bias |
| `pinn` | Sine network, first and second input derivatives, parameter gradients, and an SGD step |

The `quick` suite has one medium shape per workload. The `full` suite has small,
medium, and large shapes. The default seed is `20260810`, and every adapter
receives copies of the same generated arrays.

## Setup

Install ORMI using the instructions in the [project README](../../README.md).
PyTorch and TensorFlow are optional; use their official platform-specific
instructions:

- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)

Run from the `ormi-core` directory:

```powershell
python -m benchmarks.frameworks.run `
  --suite quick `
  --frameworks ormi pytorch tensorflow `
  --tiers direct optimized `
  --label windows-msvc-openblas-sleef `
  --json .benchmarks/framework-quick.json
```

Use `--workloads` and `--scales` to select part of the matrix. Use `--list` to
inspect the exact cases without importing a framework. A missing framework,
device, or compiler is reported under `Unavailable cases`; it does not become
a zero or an invented timing.

`--label` records the ORMI build and machine variant alongside the result. Use
a precise label whenever the result will be compared or cited. Include the
platform, compiler, and enabled performance libraries, such as
`windows-msvc-openblas-sleef` or `linux-gcc-builtin-math`. The JSON also records
the machine architecture, processor string, Python executable, framework
versions, UTC timestamp, selected cases, thread count, JIT cache path, and
explicitly configured ORMI JIT compiler.

## Modes

`direct` means ORMI graph construction and realization, PyTorch eager
execution, or TensorFlow eager execution. `optimized` means a reusable ORMI
`Plan`, `torch.compile(fullgraph=True)`, or a TensorFlow function with XLA JIT
compilation.

These modes execute the same mathematics, but their compilers and execution
models are not identical. Compare each mode by name instead of treating all
three optimized modes as the same implementation.

## Timing

The runner reports:

- `prepare`: tensor creation and construction of the callable or plan.
- `first`: the first execution after preparation.
- `median`, `p10`, and `p90`: warmed execution samples.
- ORMI CPU JIT cache entries and bytes created by each workload.

ORMI receives a new JIT cache directory for every invocation unless
`--jit-cache PATH` is supplied. Its direct first execution therefore includes
kernel compilation, while plan compilation can occur during `prepare`.
PyTorch and TensorFlow commonly defer compilation until `first`.

For isolated first-use measurements, run the tiers in separate processes:

```powershell
python -m benchmarks.frameworks.run --suite full --tiers direct
python -m benchmarks.frameworks.run --suite full --tiers optimized
```

Running both tiers together is useful for warmed throughput, but they share
process initialization and may share framework caches. Do not label those
`prepare` or `first` values as fully cold.

Select one workload and one scale when measuring the number and cost of kernels
needed by one ORMI plan. For example:

```powershell
python -m benchmarks.frameworks.run `
  --suite full `
  --workloads pinn `
  --scales medium `
  --frameworks ormi `
  --tiers optimized `
  --rounds 1 `
  --calls-per-round 1 `
  --warmup 1
```

The runner sets ORMI, OpenMP, MKL, OpenBLAS, PyTorch, and TensorFlow CPU thread
counts from `--threads`. It synchronizes accelerator work before stopping each
timer. Complete outputs, gradients, and updated parameters are compared before
warmup; compact observations are stored in JSON.

Results describe the recorded software versions, processor or GPU, compiler,
thread count, and workload shapes. They do not establish performance on other
machines. Keep the JSON artifacts when comparing architectures.

## Tests

The corpus structure and numerical-validation rules do not require PyTorch or
TensorFlow:

```powershell
python -m unittest discover -s benchmarks/frameworks/tests -v
```
