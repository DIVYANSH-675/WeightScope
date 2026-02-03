"""Portable, standard-library experiment artifact utilities."""

import json
from pathlib import Path


def write_json(path, value):
    """Validate the entire payload before creating an exclusive UTF-8 output file."""
    payload = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
