from __future__ import annotations

import argparse
import math

import numpy as np

import ormi_core as oc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small physics-informed neural network."
    )
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def init_parameters(
    width: int,
    device: str,
    rng: np.random.Generator,
) -> list[oc.Tensor]:
    dtype = oc.float64
    input_weights = oc.tensor(rng.normal(0.0, 1.0, width), dtype, device)
    hidden_biases = oc.tensor(rng.normal(0.0, 0.2, width), dtype, device)
    output_weights = oc.tensor(rng.normal(0.0, 0.3, width), dtype, device)
    output_bias = oc.tensor(0.0, dtype, device)
    return [input_weights, hidden_biases, output_weights, output_bias]


def model(x: oc.Tensor, parameters: list[oc.Tensor]) -> oc.Tensor:
    input_weights, hidden_biases, output_weights, output_bias = parameters
    hidden = (x * input_weights + hidden_biases).sin()
    return (hidden * output_weights).sum() + output_bias


def pinn_loss(
    parameters: list[oc.Tensor],
    collocation_points: np.ndarray,
    device: str,
) -> oc.Tensor:
    dtype = oc.float64
    residual_loss = oc.tensor(0.0, dtype, device)

    # The governing equation is u''(x) + u(x) = 0.
    for point in collocation_points:
        x = oc.tensor(float(point), dtype, device)
        prediction = model(x, parameters)
        residual = oc.diff(prediction, x, order=2) + prediction
        residual_loss = residual_loss + residual * residual

    residual_loss = residual_loss / len(collocation_points)

    # The initial conditions select u(x) = sin(x) from all ODE solutions.
    x0 = oc.tensor(0.0, dtype, device)
    u0 = model(x0, parameters)
    du0 = oc.diff(u0, x0)
    initial_loss = u0 * u0 + (du0 - 1.0) * (du0 - 1.0)

    return residual_loss + 10.0 * initial_loss


def update_parameters(
    parameters: list[oc.Tensor],
    gradients: list[oc.Tensor],
    learning_rate: float,
) -> list[oc.Tensor]:
    updated = [
        oc.sgd(parameter, gradient, learning_rate)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]

    # Rebuilding leaves after each step prevents
    # the training graph from retaining every previous epoch.
    return [
        oc.tensor(parameter.numpy(), parameter.dtype, parameter.device)
        for parameter in updated
    ]


def train(args: argparse.Namespace) -> list[oc.Tensor]:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.points < 1:
        raise ValueError("--points must be positive")
    if args.width < 1:
        raise ValueError("--width must be positive")
    if args.report_every < 1:
        raise ValueError("--report-every must be positive")

    rng = np.random.default_rng(args.seed)
    parameters = init_parameters(args.width, args.device, rng)
    collocation_points = np.linspace(
        0.0,
        math.pi / 2.0,
        args.points + 2,
        dtype=np.float64,
    )[1:-1]

    for epoch in range(1, args.epochs + 1):
        loss = pinn_loss(parameters, collocation_points, args.device)
        gradients = oc.grad(loss, parameters)

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:5d}  loss {loss.item():.8e}")

        parameters = update_parameters(parameters, gradients, args.learning_rate)

    return parameters


def print_solution(parameters: list[oc.Tensor], device: str) -> None:
    print("\n       x      predicted          exact      abs error")
    for point in np.linspace(0.0, math.pi / 2.0, 6):
        x = oc.tensor(float(point), oc.float64, device)
        predicted = float(model(x, parameters).item())
        exact = math.sin(point)
        error = abs(predicted - exact)
        print(
            f"{point:8.4f}  {predicted:13.8f}  "
            f"{exact:13.8f}  {error:13.3e}"
        )


def main() -> None:
    args = parse_args()
    parameters = train(args)
    print_solution(parameters, args.device)


if __name__ == "__main__":
    main()
