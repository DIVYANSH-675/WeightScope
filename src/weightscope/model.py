"""Small attention models and descriptors invariant to two attention basis changes."""

import math

import numpy as np
import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary=12, width=12, heads=2, head_width=6):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.q = nn.Parameter(torch.randn(heads, width, head_width) / math.sqrt(width))
        self.k = nn.Parameter(torch.randn_like(self.q) / math.sqrt(width))
        self.v = nn.Parameter(torch.randn_like(self.q) / math.sqrt(width))
        self.o = nn.Parameter(torch.randn(heads, head_width, width) / math.sqrt(head_width))
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.Linear(24, width))
        self.classifier = nn.Linear(width, 2)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        q, k, v = [torch.einsum("bnd,hdk->bhnk", embedded, w) for w in (self.q, self.k, self.v)]
        attention = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
        mixed = torch.einsum("bhnk,hkd->bnd", attention @ v, self.o)
        hidden = self.norm(embedded + mixed)
        return self.classifier((hidden + self.ff(hidden)).mean(1))


def descriptors(q, k, v, o):
    """Per-head QK^T and VO. These do not depend on internal head bases."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape [checkpoints, heads, width, head_width]")
    if min(q.shape) < 1:
        raise ValueError("Attention weight axes must be nonempty")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all() for value in (q, k, v, o)
    ):
        raise ValueError("Attention weights must be finite floating-point tensors")
    if o.shape != (*q.shape[:2], q.shape[3], q.shape[2]):
        raise ValueError("o must have shape [checkpoints, heads, head_width, width]")
    return torch.cat([(q @ k.transpose(-1, -2)).flatten(2), (v @ o).flatten(2)], -1)


class WeightRegressor(nn.Module):
    def __init__(self, features, width=32):
        super().__init__()
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.head_encoder = nn.Sequential(
            nn.Linear(features, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.readout = nn.Sequential(
            nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def fit_normalizer(self, train):
        if train.ndim != 3 or min(train.shape) < 1 or train.shape[-1] != len(self.center):
            raise ValueError("Expected nonempty [checkpoints,heads,features] training descriptors")
        if not train.is_floating_point() or not torch.isfinite(train).all():
            raise ValueError("Training descriptors must be finite floating-point values")
        with torch.no_grad():
            self.center.copy_(train.mean((0, 1)))
            self.scale.copy_(train.std((0, 1), unbiased=False).clamp_min(1e-5))

    def forward(self, features):
        hidden = self.head_encoder((features - self.center) / self.scale)
        return self.readout(hidden.mean(1)).squeeze(-1)


def group_split(run_ids, seed=0):
    """Split whole training runs, never individual snapshots of a shared run."""
    run_ids = np.asarray(run_ids)
    if run_ids.ndim != 1 or run_ids.size == 0 or run_ids.dtype.kind not in "iufUS":
        raise ValueError(
            "run_ids must be a nonempty one-dimensional array of numeric or string IDs"
        )
    if run_ids.dtype.kind in "iuf" and not np.isfinite(run_ids).all():
        raise ValueError("run_ids must not contain NaN or infinity")
    groups = np.unique(run_ids)
    if len(groups) < 5:
        raise ValueError("At least five independent run IDs are required")
    groups = np.random.default_rng(seed).permutation(groups)
    count = max(1, len(groups) // 5)
    partitions = (groups[2 * count :], groups[:count], groups[count : 2 * count])
    return tuple(np.flatnonzero(np.isin(run_ids, part)) for part in partitions)


def make_bank(count=20, steps=20, seed=0):
    if count < 5 or steps < 1:
        raise ValueError("count >= 5 and steps >= 1 are required")
    data_rng = torch.Generator().manual_seed(seed)
    tokens = torch.randint(12, (256, 6), generator=data_rng)
    # A permutation-invariant task matches this deliberately position-free Transformer.
    labels = (tokens.float().mean(1) > 5.5).long()
    bank = {key: [] for key in ("q", "k", "v", "o", "accuracy", "run_id")}
    for run in range(count):
        torch.manual_seed(seed + 1000 + run)
        model = TinyTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.001, 0.005, 0.02)[run % 3])
        for step in range(steps):
            ids = torch.randperm(192)[:32]
            loss = nn.functional.cross_entropy(model(tokens[ids]), labels[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if step in {0, steps - 1}:
                with torch.no_grad():
                    accuracy = (
                        (model(tokens[192:]).argmax(-1) == labels[192:]).float().mean().item()
                    )
                for key in ("q", "k", "v", "o"):
                    bank[key].append(getattr(model, key).detach().numpy().copy())
                bank["accuracy"].append(accuracy)
                bank["run_id"].append(run)
    return {key: np.asarray(value) for key, value in bank.items()}


def load_bank(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("q", "k", "v", "o", "accuracy", "run_id")}
    weights = [torch.as_tensor(arrays[key], dtype=torch.float32) for key in ("q", "k", "v", "o")]
    features = descriptors(*weights)
    accuracy = torch.as_tensor(arrays["accuracy"], dtype=torch.float32)
    if accuracy.shape != (len(features),) or arrays["run_id"].shape != (len(features),):
        raise ValueError("accuracy and run_id must contain one scalar per checkpoint")
    if not torch.isfinite(features).all() or not torch.isfinite(accuracy).all():
        raise ValueError("Bank contains non-finite values")
    if (accuracy < 0).any() or (accuracy > 1).any():
        raise ValueError("Accuracy must be in [0, 1]")
    return features, accuracy, arrays["run_id"]


def prediction_rows(prediction, target, run_ids):
    """Keep each predicted checkpoint tied to its run and observed accuracy."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    run_ids = np.asarray(run_ids)
    if (
        prediction.ndim != 1
        or prediction.shape != target.shape
        or prediction.shape != run_ids.shape
    ):
        raise ValueError("Predictions, targets, and run IDs must have matching vector shapes")
    return [
        dict(
            index=index,
            run_id=run_id,
            prediction=float(pred),
            target=float(actual),
            error=float(pred - actual),
        )
        for index, (pred, actual, run_id) in enumerate(zip(prediction, target, run_ids.tolist()))
    ]


def regression_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.ndim != 1 or not len(prediction) or prediction.shape != target.shape:
        raise ValueError("Predictions and targets must be matching nonempty vectors")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Predictions and targets must be finite")
    error = prediction - target
    return dict(
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        bias=float(error.mean()),
    )
