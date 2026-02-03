import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .io import seed_all, start_run, write_json
from .model import WeightRegressor, group_split, load_bank, make_bank


def main():
    p = argparse.ArgumentParser(
        description="Predict tiny Transformer quality from attention-weight invariants"
    )
    p.add_argument("--version", action="version", version="weightscope " + __version__)
    p.add_argument("command", choices=["bank", "train", "evaluate"])
    p.add_argument("--data")
    p.add_argument("--checkpoint")
    p.add_argument("--output", default=None)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    a.output = a.output or (
        "runs/report.json"
        if a.command == "evaluate"
        else "runs/data.npz"
        if a.command in ("generate", "bank")
        else "runs/train"
    )
    if min(a.steps, a.epochs) < 1 or a.count < 5:
        p.error("count >= 5 and positive steps/epochs are required")
    seed_all(a.seed)
    if a.command == "bank":
        target = Path(a.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez(stream, **make_bank(a.count, a.steps, a.seed))
        print(json.dumps(dict(bank=str(target), independent_runs=a.count)))
        return
    if not a.data:
        p.error("train/evaluate require --data")
    x, y, run_ids = load_bank(a.data)
    model = WeightRegressor(x.shape[-1])
    if a.command == "evaluate":
        if not a.checkpoint:
            p.error("evaluate requires --checkpoint")
        state = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            prediction = model(x)
        report = dict(
            mae=float((prediction - y).abs().mean()),
            training_mean_baseline_mae=float((y - state["training_mean"]).abs().mean()),
            count=len(y),
        )
        write_json(a.output, report)
        print(json.dumps(report))
        return
    output = start_run(a.output, vars(a))
    train, valid, test = group_split(run_ids, a.seed)
    model.fit_normalizer(x[train])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    history = []
    best = float("inf")
    training_mean = float(y[train].mean())
    for epoch in range(a.epochs):
        model.train()
        loss = (model(x[train]) - y[train]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float((model(x[valid]) - y[valid]).abs().mean())
        history.append(dict(epoch=epoch, train_mse=float(loss.detach()), validation_mae=validation))
        if validation < best:
            best = validation
            torch.save(
                dict(model=model.state_dict(), training_mean=training_mean, config=vars(a)),
                output / "best.pt",
            )
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True)["model"])
    with torch.no_grad():
        report = dict(
            test_mae=float((model(x[test]) - y[test]).abs().mean()),
            test_training_mean_baseline_mae=float((y[test] - training_mean).abs().mean()),
            train_count=len(train),
            validation_count=len(valid),
            test_count=len(test),
        )
    write_json(output / "metrics.json", history)
    write_json(output / "test.json", report)
    write_json(
        output / "split.json",
        {
            name: dict(indices=ids.tolist(), run_ids=np.unique(run_ids[ids]).tolist())
            for name, ids in zip(("train", "validation", "test"), (train, valid, test))
        },
    )
    print(json.dumps(report))
