"""Reproducible local experiment artifacts."""

import json
import platform
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def start_run(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        config, python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__
    )
    with (output / "config.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return output


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
