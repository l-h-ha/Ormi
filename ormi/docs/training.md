# Training

Install `ormi` using the [package guide](../README.md#install).
Run commands beginning with `uv` from the repository root after source setup.

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

`fit` copies batches, runs Plans, and reports progress and loss. It returns
`{epoch: loss}` for reported epochs, numbered from one. The example trains on
every sample, including the short final batch. No manual loop or tqdm import is needed.

`loss` receives one Tensor per NumPy array and returns a Tensor containing one
floating-point value.
It builds the whole loss, including model calls. You can pass several inputs,
targets, or other per-sample arrays, with any trailing dimensions. A physics loss
can use `partial(model(points), points, order=2)`. If points affect each other,
derivatives include those effects within the batch. Smaller batches cannot
reproduce a loss that depends on interactions across the whole dataset.

`update(parameter, gradient)` returns the parameter's next value as a lazy
Tensor expression. The example uses SGD. Both callbacks run only when building
the graph for a new batch shape. Python counters, tensor reads, or branches on
tensor values inside them will not run again for later batches of that shape.

Batches of the same shape reuse their input tensors and Plan. A shorter final
batch gets its own Plan on first use, after the previous batch runs. Both Plans
share model state. Set `drop_last=True` to skip the short batch.
`fit` calls `model.train()` and leaves it in training mode. All model parameters
and buffers must use one device. Inputs are copied there without changing dtype.
Supported NumPy dtypes are float16, float32, float64, int32, and bool in the
machine's byte order.

Shuffling is enabled by default. A `seed` gives a reproducible sequence of
reshuffles across epochs. Pass one
NumPy `Generator` as `rng` to continue a shuffle sequence across calls. Do not
pass both `rng` and `seed`. Neither option is accepted with `shuffle=False`.

By default every epoch is reported. `report_every=100` reads loss on epochs 1,
100, 200, and so on, plus the final epoch. Other epochs run without loss reads.
The reported loss is `sum(batch_size * batch_loss) / sum(batch_size)`. `fit`
uses the loss computed before each batch update. For batches of size 2 and 1
with losses 3 and 9, it reports `(2 * 3 + 1 * 9) / 3 = 5`. The same formula
applies when the loss sums values or ignores targets.

`verbose=1` selects a terminal/notebook progress bar, or plain epoch logs when
output is redirected. `verbose=2` always prints epoch logs. `verbose=0` is
silent. These modes do not change loss reads, updates, or random draws. The bar
covers all batches and shows the current epoch without extra tensor reads.

Do not train the same model in overlapping calls or add or replace its parameters,
buffers, or layers during training. Keep its mode and input shapes and dtypes fixed.
Errors stop training and close the progress bar. Completed updates are not undone.
Call `model.eval()` before evaluation. For custom scheduling, mixed-precision loss
scaling, manual CUDA recording, or other execution control, use the lower-level
Plan and batching APIs directly.

## Minibatches

Build the training graph and Plan once, then copy each batch into its input tensors:

```python
import numpy as np
import ormi as om
from ormi import nn

rng = np.random.default_rng(7)
values = rng.normal(size=(100, 3)).astype(np.float32)
labels = rng.integers(0, 4, size=100, dtype=np.int32)
inputs = om.tensor(np.zeros((32, 3), dtype=np.float32))
targets = om.tensor(np.zeros(32, dtype=np.int32), dtype=om.int32)
model = nn.Linear(3, 4, seed=7)
loss = nn.cross_entropy(model(inputs), targets)
parameters = model.parameters()
gradients = om.grad(loss, parameters)
updates = [
    (parameter, om.sgd(parameter, gradient, 0.01))
    for parameter, gradient in zip(parameters, gradients, strict=True)
]
plan = om.Plan([loss], updates)

for batch_inputs, batch_targets in om.batches(
    values, labels, batch_size=32, shuffle=True, rng=rng, drop_last=True,
):
    inputs.copy_from(batch_inputs)
    targets.copy_from(batch_targets)
    plan.run()
```

`batches(*arrays, batch_size=..., shuffle=False, seed=None, rng=None, drop_last=False)`
returns an iterator that can be used once. Each batch is a tuple, even for one
input. Inputs must be NumPy arrays with the same length along axis 0. All other
dimensions and dtypes stay unchanged. Strided, read-only, and memory-mapped
arrays work. Convert tensors to NumPy explicitly before calling it.

Unshuffled batches are views, so writing to one can change the source. Shuffling
makes one permutation and applies the same indices to every array, returning
copies without changing the sources. A fixed `seed` repeats the order on each
call. For different, reproducible orders across epochs, create one NumPy
`Generator` and pass it as `rng` each epoch. Do not pass both `seed` and `rng`,
or supply either without `shuffle=True`. Repeating an order requires the same
NumPy version and generator algorithm. NumPy's global random state is unchanged.

`batches()` checks arguments when you call it and shuffles when iteration starts.
An empty dataset, or one smaller than `batch_size` with `drop_last=True`,
yields no batches and leaves `rng` unchanged. Call `batches()` again for each
epoch. Do not resize the source arrays while iterating.

By default the final batch is kept, including when it is smaller than
`batch_size`. `drop_last=True` explicitly discards a shorter tail. There is no
padding. The loop above selects full batches only, so every batch fits one Plan.

`copy_from` returns `None`. It writes into a realized, contiguous leaf without
changing its shape, dtype, allocation, or graph identity. The source must have
exactly the same shape and dtype. Any number of leading dimensions is supported.
NumPy arrays can be strided or read-only. Tensor sources must already be realized
and contiguous. Copies between CPU and CUDA tensors are supported. Transfers
finish before returning, so temporary source arrays and tensors are safe to use.
Do not write to a source or destination concurrently with a copy or execution.

Copying adds no derivative relationship and runs no graph. Existing results
stay unchanged until the Plan runs again. Derivative graphs, including higher
orders, read the new inputs on each run. On CUDA, the same copies
can precede `replay.run()` after `replay = plan.record()`.

A shorter final batch needs a separate Plan. Build it after the preceding batch
runs, then reuse it in later epochs. Plans can share model parameters and buffers.
The complete example uses at most two Plans, selected by batch length because
all other shapes and dtypes stay fixed. Changing training/evaluation mode
requires a new graph and Plan.

Run the complete shuffled classification example using `fit`:

```bash
uv run python ormi/examples/minibatch.py
uv run python ormi/examples/minibatch.py --samples 257 --batch-size 32
uv run python ormi/examples/minibatch.py --samples 257 --batch-size 32 --drop-last
```

Examples display progress bars in terminals and supported notebooks. Output
redirected to a file uses plain-text loss reports. `--no-progress`
also selects plain text. The PINN example's `--report-every` controls loss
sampling. The bar does not evaluate tensors between those reports.

## Model Checkpoints

```python
om.save(model, "model.npz")
om.load(model, "model.npz")
plan.run()
```

Create the model before loading. Files contain named parameters and registered
buffers, including BatchNorm running statistics. Names, shapes, dtypes, and shared
tensor relationships must match. CPU and CUDA models can load the same file.
Loading writes into existing tensor storage. Computed outputs keep their previous
values until the Plan or CUDA replay runs again. Neither saving
nor loading runs forward or derivative graphs.

`model.state()` returns a new dictionary referencing the model's live tensors.
A shared tensor appears under each registered path and is stored
only once in the file. Built-in names include `weight`, `bias`, `running_mean`,
and `running_var`. Sequential prefixes these with child indices, such as `0.weight`.
For custom modules, pass a local name when registering state or a child:

```python
self.register_parameter(self.weight, name="weight")
self.register_buffer(self.count, name="count")
self.register_module(self.layers, name="layers")
```

Names must be unique within a module and cannot contain `.`, `/`, `\\`, or null
bytes. Unnamed registrations remain supported, using `parameter_N`, `buffer_N`,
and numeric child names. Use explicit names when registration order may change.

Files use versioned `.npz` arrays and JSON metadata, without pickle. Loading checks
every entry before changing tensors. A later copy failure can still leave partial
updates. Saving writes and flushes a temporary file, then replaces the destination.
State must be contiguous. Do not train or change state while saving or loading.
Allow host RAM for the unique tensor values and file buffers, even for CUDA models.

Files store model values. Resuming training exactly requires saving additional
state separately. These files omit model
structure, layer settings, mode, optimizer state, random-generator/Dropout counters,
epoch, and shuffle progress. They do not store graphs, Plans, or compiled kernels,
or create a model when loaded.

```bash
uv run python ormi/examples/minibatch.py --save model.npz
uv run python ormi/examples/minibatch.py --load model.npz
```

The second command starts a new training loop with the saved model values.
