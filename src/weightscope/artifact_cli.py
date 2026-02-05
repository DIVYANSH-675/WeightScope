"""Command-line utilities for saved experiment artifacts."""

import argparse
import json
from pathlib import Path

from .artifacts import build_manifest, read_records, summarize_metrics, verify_manifest, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Inspect reproducibility artifacts without loading model weights"
    )
    parser.add_argument("command", choices=["manifest", "verify", "summary"])
    parser.add_argument("--input")
    parser.add_argument("--inputs", nargs="+", default=[])
    parser.add_argument("--base", default=".")
    parser.add_argument("--metadata", default="{}", help="JSON object stored with a manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command not in ("manifest", "compare") and not args.input:
        parser.error(args.command + " requires --input")
    try:
        if args.command == "manifest":
            report = build_manifest(args.inputs, args.base, json.loads(args.metadata))
        elif args.command == "verify":
            report = verify_manifest(
                json.loads(Path(args.input).read_text(encoding="utf-8")), args.base
            )
        elif args.command == "summary":
            report = summarize_metrics(read_records(args.input))
        write_json(args.output, report)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False))
    if args.command == "verify" and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
