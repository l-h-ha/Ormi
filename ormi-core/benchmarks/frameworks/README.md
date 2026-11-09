# Framework benchmarks

ORMI, PyTorch, and TensorFlow get copies of the same NumPy inputs. The runner
times setup, the first call, and calls after warmup separately.

## Workloads

| Workload | Timed work |
| --- | --- |
| `pointwise` | `(input * scale + bias) * (input - bias) + input * 0.125`, with broadcasting |
| `reduction` | Square `input * row + column`, then take its mean over axes 0 and 2 |
| `dense` | Mean squared error of `input @ weight + bias` against `target`, and gradients for weight and bias |
| `pinn` | Sine network, first and second input derivatives, parameter gradients, and an SGD step |

The `quick` suite has one medium shape per workload. The `full` suite has small,
medium, and large shapes. The default seed is `20260810`.

The ORMI PINN uses `nn.Sequential`, `nn.Linear`, and `nn.Sin`. Its layers take
the shared parameter arrays directly, without drawing new initial weights.
`model.parameters()` lists updates in weight, bias, weight, bias order.

## Setup

Install ORMI using the instructions in the [project README](../../README.md).
PyTorch and TensorFlow are optional. Use their official instructions for your
platform:

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

Use `--workloads` and `--scales` to select cases. Use `--list` to see the cases
without importing a framework. The runner lists skipped cases under
`Unavailable cases` and omits their timings.

`--label` records the ORMI build and machine with the result. Include the
platform, compiler, and enabled performance libraries, such as
`windows-msvc-openblas-sleef` or `linux-gcc-builtin-math`. The JSON also records
the machine architecture, processor string, Python executable, framework
versions, UTC timestamp, selected cases, thread count, JIT cache path, and
explicitly configured ORMI JIT compiler.

## Modes

In `direct` mode, ORMI rebuilds and executes its lazy graph on every call.
PyTorch and TensorFlow use eager execution. In `optimized` mode, the runner
reuses an ORMI `Plan`, a `torch.compile(fullgraph=True)` callable, or a
TensorFlow function with XLA JIT compilation.

Compare results by framework and mode.

Without a C++ compiler, both ORMI CPU modes can use cached fused kernels and
fall back to precompiled kernels. To time only the precompiled path, start
with an empty JIT cache and set `ORMI_CPU_JIT_CXX` to
`ormi-compiler-does-not-exist`. Record whether the compiler was available and
whether the cache started empty.

## Timing

The runner reports:

- `prepare`: tensor creation and construction of the callable or plan.
- `first`: the first execution after preparation.
- `median`, `p10`, and `p90`: median, 10th percentile, and 90th percentile of
  warmed per-call execution times. Each sample is one round's elapsed time
  divided by `--calls-per-round`.
- ORMI CPU JIT cache entries and bytes created by each workload.

ORMI receives a new CPU JIT cache directory for every run unless
`--jit-cache PATH` is supplied. The first direct run may compile CPU kernels.
Plans may compile during `prepare`. PyTorch and TensorFlow usually compile
during `first` in optimized mode.
The runner does not reset the CUDA PTX disk cache.

To measure first-use costs separately, run each mode in its own process:

```powershell
python -m benchmarks.frameworks.run --suite full --tiers direct
python -m benchmarks.frameworks.run --suite full --tiers optimized
```

Running both tiers together is useful for warmed throughput, but they share
process initialization and may share framework caches. Do not label those
`prepare` or `first` values as fully cold.

To count the CPU JIT cache entries added by one ORMI Plan, select one workload
and one scale:

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
timer. All outputs, gradients, and updated parameters are compared after
the timed first execution and before warmup. JSON results include output means
and ranges, gradient means and L2 norms, or loss and parameter means, depending
on the workload.

Results describe the recorded software versions, processor or GPU, compiler,
thread count, and workload shapes. They do not establish performance on other
machines. Keep the JSON files when comparing architectures.

## Tests

Workload, input, comparison, and option tests run without PyTorch or TensorFlow:

```powershell
python -m unittest discover -s benchmarks/frameworks/tests -v
```
