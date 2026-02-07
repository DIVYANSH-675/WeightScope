"""Command-line utilities for saved experiment artifacts."""

import argparse
import json
from pathlib import Path

from .artifacts import (
    bootstrap_mean,
    build_manifest,
    compare_metrics,
    export_csv,
    read_records,
    summarize_metrics,
    verify_manifest,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect reproducibility artifacts without loading model weights"
    )
    parser.add_argument(
        "command", choices=["manifest", "verify", "summary", "csv", "compare", "bootstrap"]
    )
    parser.add_argument("--input")
    parser.add_argument("--inputs", nargs="+", default=[])
    parser.add_argument("--base", default=".")
    parser.add_argument("--metadata", default="{}", help="JSON object stored with a manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--key")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
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
        elif args.command == "csv":
            export_csv(read_records(args.input), args.output)
            print(json.dumps(dict(output=args.output)))
            return
        elif args.command == "compare":
            if not args.before or not args.after:
                parser.error("compare requires --before and --after")
            report = compare_metrics(read_records(args.before), read_records(args.after))
        elif args.command == "bootstrap":
            if not args.key:
                parser.error("bootstrap requires --key")
            rows = read_records(args.input)
            values = []
            for row in rows:
                if args.key not in row:
                    raise ValueError("Every record must contain the selected bootstrap metric")
                values.append(row[args.key])
            report = bootstrap_mean(values, args.samples, args.confidence, args.seed)
        write_json(args.output, report)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False))
    if args.command == "verify" and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
