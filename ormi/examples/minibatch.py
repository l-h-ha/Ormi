from __future__ import annotations

import argparse
import numpy as np

import ormi as om
from ormi import nn

###
###
###

def train(
    epochs: int = 30,
    device: str = "cpu",
    seed: int = 7,
    *,
    load_path: str | None = None,
    save_path: str | None = None,
    progress: bool = True,
    samples: int = 256,
    batch_size: int = 32,
    drop_last: bool = False,
) -> nn.Module:

    if epochs < 1:
        raise ValueError("epochs must be positive")

    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")

    if drop_last and samples < batch_size:
        raise ValueError("drop_last would discard every sample")

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(samples, 3)).astype(np.float32)
    labels = np.argmax(values, axis=-1).astype(np.int32)

    model = nn.Sequential([
        nn.Linear(3, 16, device=device, seed=seed),
        nn.Tanh(),
        nn.Linear(16, 3, device=device, seed=seed + 1),
    ])

    if load_path is not None:
        om.load(model, load_path)

    def loss(inputs: om.Tensor, targets: om.Tensor) -> om.Tensor:
        return nn.cross_entropy(model(inputs), targets)

    om.fit(
        model, values, labels, loss=loss,
        update=lambda parameter, gradient: om.sgd(parameter, gradient, 0.1),
        epochs=epochs, batch_size=batch_size, rng=rng, drop_last=drop_last,
        verbose=1 if progress else 2,
    )

    model.eval()
    scores = model(om.tensor(values, device=device)).numpy()
    accuracy = np.mean(np.argmax(scores, axis=-1) == labels)

    print(f"training accuracy {accuracy:.1%}")

    if save_path is not None:
        om.save(model, save_path)

    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier with one reusable Plan per batch shape.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", dest="save_path")
    parser.add_argument("--load", dest="load_path")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--drop-last", action="store_true")
    args = parser.parse_args()

    train(
        args.epochs, args.device, args.seed, load_path=args.load_path, save_path=args.save_path,
        progress=not args.no_progress,
        samples=args.samples, batch_size=args.batch_size, drop_last=args.drop_last,
    )

###
###
###

if __name__ == "__main__":
    main()
