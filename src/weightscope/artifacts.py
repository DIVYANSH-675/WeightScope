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


def file_digest(path, chunk_size=1024 * 1024):
    """Stream a file into SHA-256 without loading it completely into memory."""
    import hashlib

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
