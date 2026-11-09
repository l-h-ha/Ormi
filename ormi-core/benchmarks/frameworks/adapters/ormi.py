from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import ormi_core as oc
from ormi_core import nn

from ..model import (
    Case,
    Data,
    Observation,
    Spec,
    Tier,
    gradient_observation,
    output_observation,
    step_observation,
)
from ..workloads import LEARNING_RATE

###
###
###

def configure(threads: int) -> None:
    del threads

def version() -> str:
    return oc.__version__

def _dtype(name: str) -> oc.DType:
    return {
        "float16": oc.float16,
        "float32": oc.float32,
        "float64": oc.float64,
    }[name]

def _case(
    tier: Tier,
    mode: str,
    start: int,
    run: Callable[[], None],
    values: Callable[[], tuple[np.ndarray, ...]],
    observe: Callable[[], Observation],
    device: oc.Device,
) -> Case:
    return Case(
        framework="ORMI",
        tier=tier,
        mode=mode,
        prepare_ms=(time.perf_counter_ns() - start) / 1e6,
        run=run,
        values=values,
        observe=observe,
        synchronize=lambda: oc.synchronize(device),
    )

def _pointwise(spec: Spec, data: Data, tier: Tier, device: oc.Device) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = oc.tensor(data.values["input"], dtype, device)
    scale = oc.tensor(data.values["scale"], dtype, device)
    bias = oc.tensor(data.values["bias"], dtype, device)

    def graph() -> oc.Tensor:
        transformed = input_ * scale + bias
        return transformed * (input_ - bias) + input_ * 0.125

    last: oc.Tensor | None = None

    if tier == "direct":
        def run() -> None:
            nonlocal last
            last = graph()
            last.realize()

        mode = "direct"
    else:
        last = graph()
        plan = oc.Plan([last])

        def run() -> None:
            plan.run()

        mode = "plan"

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("ORMI pointwise workload has not run.")

        return (np.asarray(last.numpy()),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, mode, start, run, values, observe, device)

def _reduction(spec: Spec, data: Data, tier: Tier, device: oc.Device) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = oc.tensor(data.values["input"], dtype, device)
    row = oc.tensor(data.values["row"], dtype, device)
    column = oc.tensor(data.values["column"], dtype, device)

    def graph() -> oc.Tensor:
        transformed = input_ * row + column
        return (transformed * transformed).mean((0, 2))

    last: oc.Tensor | None = None

    if tier == "direct":
        def run() -> None:
            nonlocal last
            last = graph()
            last.realize()

        mode = "direct"
    else:
        last = graph()
        plan = oc.Plan([last])

        def run() -> None:
            plan.run()

        mode = "plan"

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("ORMI reduction workload has not run.")

        return (np.asarray(last.numpy()),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, mode, start, run, values, observe, device)

def _dense(spec: Spec, data: Data, tier: Tier, device: oc.Device) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = oc.tensor(data.values["input"], dtype, device)
    target = oc.tensor(data.values["target"], dtype, device)
    weight = oc.tensor(data.values["weight"], dtype, device)
    bias = oc.tensor(data.values["bias"], dtype, device)
    parameters = [weight, bias]

    def graph() -> tuple[oc.Tensor, list[oc.Tensor]]:
        error = input_ @ weight + bias - target
        loss = (error * error).mean()
        return loss, oc.grad(loss, parameters)

    last_loss: oc.Tensor | None = None
    last_gradients: list[oc.Tensor] = []

    if tier == "direct":
        def run() -> None:
            nonlocal last_loss, last_gradients
            last_loss, last_gradients = graph()
            oc.Plan([last_loss, *last_gradients]).run()

        mode = "direct"
    else:
        last_loss, last_gradients = graph()
        plan = oc.Plan([last_loss, *last_gradients])

        def run() -> None:
            plan.run()

        mode = "plan"

    def values() -> tuple[np.ndarray, ...]:
        if last_loss is None:
            raise RuntimeError("ORMI dense workload has not run.")

        return (
            np.asarray(last_loss.numpy()),
            *(np.asarray(gradient.numpy()) for gradient in last_gradients),
        )

    def observe() -> Observation:
        current = values()
        return gradient_observation(
            current[0],
            current[1:],
        )

    return _case(tier, mode, start, run, values, observe, device)

def _pinn(spec: Spec, data: Data, tier: Tier, device: oc.Device) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = oc.tensor(data.values["input"], dtype, device)
    model = nn.Sequential([
        nn.Linear(
            oc.tensor(data.values["first_weight"], dtype, device),
            oc.tensor(data.values["first_bias"], dtype, device),
        ),
        nn.Sin(),
        nn.Linear(
            oc.tensor(data.values["second_weight"], dtype, device),
            oc.tensor(data.values["second_bias"], dtype, device),
        ),
    ])
    parameters = model.parameters()

    def graph() -> tuple[oc.Tensor, list[oc.Tensor]]:
        prediction = model(input_)
        first = oc.grad(
            prediction,
            [input_],
            oc.ones_like(prediction),
        )[0]
        second = oc.grad(first, [input_], oc.ones_like(first))[0]
        residual = second + prediction
        loss = (residual * residual).mean()
        gradients = oc.grad(loss, parameters)
        updates = [
            oc.sgd(parameter, gradient, LEARNING_RATE)
            for parameter, gradient in zip(
                parameters,
                gradients,
                strict=True,
            )
        ]
        return loss, updates

    last_loss: oc.Tensor | None = None

    if tier == "direct":
        def run() -> None:
            nonlocal last_loss
            last_loss, updates = graph()
            oc.Plan(
                [last_loss],
                list(zip(parameters, updates, strict=True)),
            ).run()

        mode = "direct"
    else:
        last_loss, updates = graph()
        plan = oc.Plan(
            [last_loss],
            list(zip(parameters, updates, strict=True)),
        )

        def run() -> None:
            plan.run()

        mode = "plan"

    def values() -> tuple[np.ndarray, ...]:
        if last_loss is None:
            raise RuntimeError("ORMI PINN workload has not run.")

        return (
            np.asarray(last_loss.numpy()),
            *(np.asarray(parameter.numpy()) for parameter in parameters),
        )

    def observe() -> Observation:
        current = values()
        return step_observation(
            current[0],
            current[1:],
        )

    return _case(tier, mode, start, run, values, observe, device)

def prepare(spec: Spec, data: Data, tier: Tier, device_name: str) -> Case:
    device = oc.Device(device_name)

    if spec.name == "pointwise":
        return _pointwise(spec, data, tier, device)

    if spec.name == "reduction":
        return _reduction(spec, data, tier, device)

    if spec.name == "dense":
        return _dense(spec, data, tier, device)

    if spec.name == "pinn":
        return _pinn(spec, data, tier, device)

    raise ValueError(f"Unknown workload: {spec.name}")
