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


def _within_base(path, base):
    base = Path(base).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as error:
        raise ValueError("Manifest files must stay inside the supplied base directory") from error
    return resolved, relative.as_posix()


def build_manifest(paths, base=".", metadata=None):
    """Describe a fixed set of files using portable relative paths and digests."""
    paths = list(paths)
    if not paths:
        raise ValueError("At least one input file is required")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("Manifest metadata must be an object")
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    files = {}
    for path in paths:
        resolved, relative = _within_base(path, base)
        files[relative] = dict(
            path=relative, sha256=file_digest(resolved), bytes=resolved.stat().st_size
        )
    return dict(schema_version=1, metadata=metadata, files=[files[key] for key in sorted(files)])


def verify_manifest(manifest, base="."):
    """Return per-file verification results; changed inputs never pass silently."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must list at least one file")
    results = []
    seen = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Invalid manifest file entry")
        path = entry["path"]
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if (
            Path(path).is_absolute()
            or path in seen
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("Invalid path, digest, or byte count in manifest")
        seen.add(path)
        resolved, _ = _within_base(path, base)
        try:
            valid = file_digest(resolved) == digest and resolved.stat().st_size == size
            status = "ok" if valid else "changed"
        except FileNotFoundError:
            status = "missing"
        results.append(dict(path=path, status=status))
    return dict(valid=all(row["status"] == "ok" for row in results), files=results)
