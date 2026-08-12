from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

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


def configure(threads: int) -> None:
    global _configured
    if _configured:
        return
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _configured = True


def version() -> str:
    return torch.__version__


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Unsupported("PyTorch cannot access a CUDA device.")
    return device


def _tensor(value: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.copy()).to(device=device, dtype=dtype)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _case(
    tier: Tier,
    start: int,
    run: Callable[[], None],
    values: Callable[[], tuple[np.ndarray, ...]],
    observe: Callable[[], Observation],
    device: torch.device,
) -> Case:
    return Case(
        framework="PyTorch",
        tier=tier,
        mode="eager" if tier == "direct" else "compile",
        prepare_ms=(time.perf_counter_ns() - start) / 1e6,
        run=run,
        values=values,
        observe=observe,
        synchronize=(
            torch.cuda.synchronize
            if device.type == "cuda"
            else lambda: None
        ),
    )


def _compile(function: Callable[..., object], tier: Tier) -> Callable[..., object]:
    return (
        function
        if tier == "direct"
        else torch.compile(function, fullgraph=True)
    )


def _pointwise(
    spec: Spec,
    data: Data,
    tier: Tier,
    device: torch.device,
) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = _tensor(data.values["input"], dtype, device)
    scale = _tensor(data.values["scale"], dtype, device)
    bias = _tensor(data.values["bias"], dtype, device)

    def expression(
        input_value: torch.Tensor,
        scale_value: torch.Tensor,
        bias_value: torch.Tensor,
    ) -> torch.Tensor:
        transformed = input_value * scale_value + bias_value
        return (
            transformed * (input_value - bias_value) +
            input_value * 0.125
        )

    execute = _compile(expression, tier)
    last: torch.Tensor | None = None

    def run() -> None:
        nonlocal last
        last = execute(input_, scale, bias)

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("PyTorch pointwise workload has not run.")
        return (_numpy(last),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, start, run, values, observe, device)


def _reduction(
    spec: Spec,
    data: Data,
    tier: Tier,
    device: torch.device,
) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = _tensor(data.values["input"], dtype, device)
    row = _tensor(data.values["row"], dtype, device)
    column = _tensor(data.values["column"], dtype, device)

    def expression(
        input_value: torch.Tensor,
        row_value: torch.Tensor,
        column_value: torch.Tensor,
    ) -> torch.Tensor:
        transformed = input_value * row_value + column_value
        return torch.mean(transformed * transformed, dim=(0, 2))

    execute = _compile(expression, tier)
    last: torch.Tensor | None = None

    def run() -> None:
        nonlocal last
        last = execute(input_, row, column)

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("PyTorch reduction workload has not run.")
        return (_numpy(last),)

    def observe() -> Observation:
        return output_observation(values()[0])

    return _case(tier, start, run, values, observe, device)


def _dense(
    spec: Spec,
    data: Data,
    tier: Tier,
    device: torch.device,
) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = _tensor(data.values["input"], dtype, device)
    target = _tensor(data.values["target"], dtype, device)
    parameters = (
        _tensor(data.values["weight"], dtype, device),
        _tensor(data.values["bias"], dtype, device),
    )

    def loss_function(
        current: tuple[torch.Tensor, torch.Tensor],
        input_value: torch.Tensor,
        target_value: torch.Tensor,
    ) -> torch.Tensor:
        weight, bias = current
        error = input_value @ weight + bias - target_value
        return torch.mean(error * error)

    value_and_grad = torch.func.grad_and_value(loss_function)

    def expression(
        current: tuple[torch.Tensor, torch.Tensor],
        input_value: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gradients, loss = value_and_grad(current, input_value, target_value)
        return loss, gradients[0], gradients[1]

    execute = _compile(expression, tier)
    last: tuple[torch.Tensor, ...] | None = None

    def run() -> None:
        nonlocal last
        last = execute(parameters, input_, target)

    def values() -> tuple[np.ndarray, ...]:
        if last is None:
            raise RuntimeError("PyTorch dense workload has not run.")
        return tuple(_numpy(value) for value in last)

    def observe() -> Observation:
        current = values()
        return gradient_observation(
            current[0],
            current[1:],
        )

    return _case(tier, start, run, values, observe, device)


def _pinn(
    spec: Spec,
    data: Data,
    tier: Tier,
    device: torch.device,
) -> Case:
    start = time.perf_counter_ns()
    dtype = _dtype(spec.dtype)
    input_ = _tensor(data.values["input"], dtype, device)
    parameters: tuple[torch.Tensor, ...] = (
        _tensor(data.values["first_weight"], dtype, device),
        _tensor(data.values["first_bias"], dtype, device),
        _tensor(data.values["second_weight"], dtype, device),
        _tensor(data.values["second_bias"], dtype, device),
    )

    def prediction(
        input_value: torch.Tensor,
        current: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        first_weight, first_bias, second_weight, second_bias = current
        hidden = torch.sin(
            input_value.reshape(1, 1) @ first_weight + first_bias
        )
        return (hidden @ second_weight + second_bias).reshape(())

    first_derivative = torch.func.grad(prediction)
    second_derivative = torch.func.grad(first_derivative)

    def loss_function(
        current: tuple[torch.Tensor, ...],
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        flat = input_value.reshape(-1)
        predictions = torch.vmap(
            prediction,
            in_dims=(0, None),
        )(flat, current)
        second = torch.vmap(
            second_derivative,
            in_dims=(0, None),
        )(flat, current)
        residual = second + predictions
        return torch.mean(residual * residual)

    value_and_grad = torch.func.grad_and_value(loss_function)

    def step(
        current: tuple[torch.Tensor, ...],
        input_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        gradients, loss = value_and_grad(current, input_value)
        updates = tuple(
            parameter - LEARNING_RATE * gradient
            for parameter, gradient in zip(
                current,
                gradients,
                strict=True,
            )
        )
        return (loss, *updates)

    execute = _compile(step, tier)
    last_loss: torch.Tensor | None = None

    def run() -> None:
        nonlocal parameters, last_loss
        result = execute(parameters, input_)
        last_loss = result[0]
        parameters = tuple(result[1:])

    def values() -> tuple[np.ndarray, ...]:
        if last_loss is None:
            raise RuntimeError("PyTorch PINN workload has not run.")
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

    return _case(tier, start, run, values, observe, device)


def prepare(spec: Spec, data: Data, tier: Tier, device_name: str) -> Case:
    device = _device(device_name)
    if spec.name == "pointwise":
        return _pointwise(spec, data, tier, device)
    if spec.name == "reduction":
        return _reduction(spec, data, tier, device)
    if spec.name == "dense":
        return _dense(spec, data, tier, device)
    if spec.name == "pinn":
        return _pinn(spec, data, tier, device)
    raise ValueError(f"Unknown workload: {spec.name}")
