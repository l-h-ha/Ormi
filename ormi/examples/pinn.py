from __future__ import annotations

import argparse
import math
import numpy as np

import ormi as om
from ormi import nn

###
###
###

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
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()

def init_model(
    width: int,
    device: str,
    seed: int,
) -> nn.Module:
    dtype = om.float64
    seeds = np.random.default_rng(seed).integers(0, 2**64, size=2, dtype=np.uint64)
    return nn.Sequential([
        nn.Linear(1, width, dtype=dtype, device=device, seed=int(seeds[0])),
        nn.Sin(),
        nn.Linear(width, 1, dtype=dtype, device=device, seed=int(seeds[1])),
    ])

def pinn_loss(
    model: nn.Module,
    collocation_points: om.Tensor,
) -> om.Tensor:
    # Equation is u''(x) + u(x) = 0.
    prediction = model(collocation_points)
    second = om.partial(prediction, collocation_points, order=2)
    residual = second + prediction
    residual_loss = nn.mse_loss(residual, om.zeros_like(residual))

    # The initial conditions select u(x) = sin(x) from all ODE solutions.
    x0 = om.tensor([[0.0]], om.float64, collocation_points.device)
    u0 = model(x0)
    du0 = om.partial(u0, x0)
    initial_loss = (
        nn.mse_loss(u0, om.zeros_like(u0)) +
        nn.mse_loss(du0, om.ones_like(du0))
    )

    return residual_loss + 10.0 * initial_loss

def train(args: argparse.Namespace) -> nn.Module:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    if args.points < 1:
        raise ValueError("--points must be positive")

    if args.width < 1:
        raise ValueError("--width must be positive")

    if args.report_every < 1:
        raise ValueError("--report-every must be positive")

    model = init_model(args.width, args.device, args.seed)
    positions = np.linspace(
        0.0,
        math.pi / 2.0,
        args.points + 2,
        dtype=np.float64,
    )[1:-1]

    om.fit(
        model, positions.reshape(-1, 1), loss=lambda points: pinn_loss(model, points),
        update=lambda parameter, gradient: om.sgd(parameter, gradient, args.learning_rate),
        epochs=args.epochs, batch_size=args.points, shuffle=False,
        report_every=args.report_every, verbose=2 if args.no_progress else 1,
    )

    return model

def print_solution(model: nn.Module, device: str) -> None:
    print("\n       x      predicted          exact      abs error")
    points = np.linspace(0.0, math.pi / 2.0, 6)
    predictions = model(om.tensor(points.reshape(-1, 1), om.float64, device)).numpy().reshape(-1)

    for point, predicted in zip(points, predictions, strict=True):
        exact = math.sin(point)
        error = abs(predicted - exact)
        print(
            f"{point:8.4f}  {predicted:13.8f}  "
            f"{exact:13.8f}  {error:13.3e}"
        )

###
###
###

def main() -> None:
    args = parse_args()
    model = train(args)
    print_solution(model, args.device)

if __name__ == "__main__":
    main()
