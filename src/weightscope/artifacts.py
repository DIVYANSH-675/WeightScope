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


def read_records(path):
    """Read JSON arrays, single records, saved rows/episodes, or streamed JSONL."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            records = (json.loads(line) for line in stream if line.strip())
        else:
            payload = json.load(stream)
            if isinstance(payload, dict):
                payload = payload.get("rows", payload.get("episodes", [payload]))
            if not isinstance(payload, list):
                raise ValueError("Metrics must contain an array or object")
            records = payload
        for row in records:
            if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
                raise ValueError("Each metric record must be an object with string keys")
            yield row


def _finite_number(value):
    import math

    if type(value) not in (int, float):
        raise ValueError("Metric values must be real numbers, excluding booleans")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("Metric values must be finite") from error
    if not math.isfinite(value):
        raise ValueError("Metric values must be finite")
    return value


def summarize_metrics(records):
    """Use Welford accumulation for numeric columns; booleans are not metrics."""
    columns = {}
    rows = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Metric records must be objects")
        rows += 1
        for key, value in record.items():
            if type(value) not in (int, float):
                continue
            value = _finite_number(value)
            stats = columns.setdefault(key, dict(count=0, mean=0.0, m2=0.0, min=value, max=value))
            stats["count"] += 1
            delta = value - stats["mean"]
            stats["mean"] += delta / stats["count"]
            stats["m2"] += delta * (value - stats["mean"])
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
    if not columns:
        raise ValueError("No numeric metrics were found")
    for stats in columns.values():
        stats["std"] = max(0.0, stats.pop("m2") / stats["count"]) ** 0.5
    return dict(rows=rows, metrics=columns)


def export_csv(records, path):
    """Export flat records with a stable union of columns and exclusive creation."""
    import csv
    import io

    records = list(records)
    if not records:
        raise ValueError("At least one record is required for CSV export")
    for record in records:
        if not isinstance(record, dict) or any(not isinstance(key, str) for key in record):
            raise ValueError("CSV records must have string keys")
        for value in record.values():
            if value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("CSV export accepts flat scalar records only")
            if type(value) in (int, float):
                _finite_number(value)
    columns = sorted({key for record in records for key in record})
    if not columns:
        raise ValueError("CSV export requires at least one column")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(buffer.getvalue())


def compare_metrics(before, after):
    """Compare column means; leave metric direction and scientific interpretation explicit."""
    left = summarize_metrics(before)["metrics"]
    right = summarize_metrics(after)["metrics"]
    common = sorted(left.keys() & right.keys())
    if not common:
        raise ValueError("Reports have no numeric metrics in common")
    differences = {}
    for key in common:
        old, new = left[key]["mean"], right[key]["mean"]
        differences[key] = dict(
            before=old,
            after=new,
            delta=new - old,
            relative_change=(new - old) / abs(old) if old else None,
        )
    return dict(
        metrics=differences,
        only_before=sorted(left.keys() - right.keys()),
        only_after=sorted(right.keys() - left.keys()),
    )
