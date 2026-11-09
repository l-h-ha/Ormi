from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import tensorflow as tf

from ..model import (
    Case,
    Data,
    Observation,
    Spec,
    Tier,
    Unsupported,
    gradient_observation,
    output_observation,
    step_observation,
)
from ..workloads import LEARNING_RATE

_configured = False

###
###
###

def configure(threads: int) -> None:
    global _configured

    if _configured:
        return

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    set_synchronous = getattr(
        tf.config.experimental,
        "set_synchronous_execution",
        None,
    )

    if set_synchronous is not None:
        set_synchronous(True)

    _configured = True

def version() -> str:
    return tf.__version__

def _dtype(name: str) -> tf.dtypes.DType:
    return {
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64,
    }[name]

def _device(name: str) -> str:
    if name.startswith("cuda"):
        if not tf.config.list_physical_devices("GPU"):
            raise Unsupported("TensorFlow cannot access a CUDA device.")

        return "/GPU:0"

    return "/CPU:0"

def _numpy(value: tf.Tensor | tf.Variable) -> np.ndarray:
    return np.asarray(value.numpy())

def _case(
    tier: Tier,
    start: int,
    run: Callable[[], None],
    values: Callable[[], tuple[np.ndarray, ...]],
    observe: Callable[[], Observation],
    latest: Callable[[], tf.Tensor | None],
) -> Case:
    def synchronize() -> None:
        wait = getattr(tf.experimental, "async_wait", None)

        if wait is not None:
            wait()
            return

        value = latest()

        if value is not None:
            value.numpy()

    return Case(
        framework="TensorFlow",
        tier=tier,
        mode="eager" if tier == "direct" else "xla",
        prepare_ms=(time.perf_counter_ns() - start) / 1e6,
        run=run,
        values=values,
        observe=observe,
        synchronize=synchronize,
    )

def _compile(function: Callable[..., object], tier: Tier) -> Callable[..., object]:
    return (
        function
        if tier == "direct"
        else tf.function(function, autograph=False, jit_compile=True)
    )

def _pointwise(spec: Spec, data: Data, tier: Tier, placement: str) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)

    with tf.device(placement):
        input_ = tf.convert_to_tensor(data.values["input"], dtype=dtype)
        scale = tf.convert_to_tensor(data.values["scale"], dtype=dtype)
        bias = tf.convert_to_tensor(data.values["bias"], dtype=dtype)

        def expression(
            input_value: tf.Tensor,
            scale_value: tf.Tensor,
            bias_value: tf.Tensor,
        ) -> tf.Tensor:
            transformed = input_value * scale_value + bias_value
            return (
                transformed * (input_value - bias_value) +
                input_value * 0.125
            )

        execute = _compile(expression, tier)

    last: tf.Tensor | None = None

    def run() -> None:
        nonlocal last
        last = execute(input_, scale, bias)

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("TensorFlow pointwise workload has not run.")

        return (_numpy(last),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, start, run, values, observe, lambda: last)

def _reduction(spec: Spec, data: Data, tier: Tier, placement: str) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)

    with tf.device(placement):
        input_ = tf.convert_to_tensor(data.values["input"], dtype=dtype)
        row = tf.convert_to_tensor(data.values["row"], dtype=dtype)
        column = tf.convert_to_tensor(data.values["column"], dtype=dtype)

        def expression(
            input_value: tf.Tensor,
            row_value: tf.Tensor,
            column_value: tf.Tensor,
        ) -> tf.Tensor:
            transformed = input_value * row_value + column_value
            return tf.reduce_mean(transformed * transformed, axis=(0, 2))

        execute = _compile(expression, tier)

    last: tf.Tensor | None = None

    def run() -> None:
        nonlocal last
        last = execute(input_, row, column)

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("TensorFlow reduction workload has not run.")

        return (_numpy(last),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, start, run, values, observe, lambda: last)

def _dense(spec: Spec, data: Data, tier: Tier, placement: str) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)

    with tf.device(placement):
        input_ = tf.convert_to_tensor(data.values["input"], dtype=dtype)
        target = tf.convert_to_tensor(data.values["target"], dtype=dtype)
        parameters = (
            tf.Variable(data.values["weight"], trainable=True, dtype=dtype),
            tf.Variable(data.values["bias"], trainable=True, dtype=dtype),
        )

        def expression() -> tuple[tf.Tensor, ...]:
            with tf.GradientTape() as tape:
                weight, bias = parameters
                error = tf.linalg.matmul(input_, weight) + bias - target
                loss = tf.reduce_mean(error * error)

            gradients = tape.gradient(loss, parameters)
            return (loss, *gradients)

        execute = _compile(expression, tier)

    last: tuple[tf.Tensor, ...] | None = None

    def run() -> None:
        nonlocal last
        last = execute()

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("TensorFlow dense workload has not run.")

        return tuple(_numpy(value) for value in last)

    def observe() -> Observation:
        current = values()
        return gradient_observation(
            current[0],
            current[1:],
        )

    return _case(
        tier,
        start,
        run,
        values,
        observe,
        lambda: None if last is None else last[0],
    )

def _pinn(spec: Spec, data: Data, tier: Tier, placement: str) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)

    with tf.device(placement):
        input_ = tf.convert_to_tensor(data.values["input"], dtype=dtype)
        parameters: tuple[tf.Variable, ...] = (
            tf.Variable(
                data.values["first_weight"],
                trainable=True,
                dtype=dtype,
            ),
            tf.Variable(
                data.values["first_bias"],
                trainable=True,
                dtype=dtype,
            ),
            tf.Variable(
                data.values["second_weight"],
                trainable=True,
                dtype=dtype,
            ),
            tf.Variable(
                data.values["second_bias"],
                trainable=True,
                dtype=dtype,
            ),
        )

        def expression() -> tf.Tensor:
            with tf.GradientTape() as parameter_tape:
                with tf.GradientTape() as second_tape:
                    second_tape.watch(input_)

                    with tf.GradientTape() as first_tape:
                        first_tape.watch(input_)
                        (
                            first_weight,
                            first_bias,
                            second_weight,
                            second_bias,
                        ) = parameters
                        hidden = tf.sin(
                            tf.linalg.matmul(input_, first_weight) +
                            first_bias
                        )
                        prediction = (
                            tf.linalg.matmul(hidden, second_weight) +
                            second_bias
                        )

                    first = first_tape.gradient(prediction, input_)

                second = second_tape.gradient(first, input_)
                residual = second + prediction
                loss = tf.reduce_mean(residual * residual)

            gradients = parameter_tape.gradient(loss, parameters)

            for parameter, gradient in zip(
                parameters,
                gradients,
                strict=True,
            ):
                parameter.assign_sub(LEARNING_RATE * gradient)

            return loss

        execute = _compile(expression, tier)

    last_loss: tf.Tensor | None = None

    def run() -> None:
        nonlocal last_loss
        last_loss = execute()

    def values() -> tuple[np.ndarray, ...]:
        if last_loss is None:
            raise RuntimeError("TensorFlow PINN workload has not run.")

        return (
            _numpy(last_loss),
            *(_numpy(parameter) for parameter in parameters),
        )

    def observe() -> Observation:
        current = values()
        return step_observation(
            current[0],
            current[1:],
        )

    return _case(
        tier,
        start,
        run,
        values,
        observe,
        lambda: last_loss,
    )

def prepare(spec: Spec, data: Data, tier: Tier, device_name: str) -> Case:
    placement = _device(device_name)

    if spec.name == "pointwise":
        return _pointwise(spec, data, tier, placement)

    if spec.name == "reduction":
        return _reduction(spec, data, tier, placement)

    if spec.name == "dense":
        return _dense(spec, data, tier, placement)

    if spec.name == "pinn":
        return _pinn(spec, data, tier, placement)

    raise ValueError(f"Unknown workload: {spec.name}")
