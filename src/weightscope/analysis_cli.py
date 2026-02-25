"""Command-line research diagnostics for saved experiment observations."""

import argparse
import json

from . import __version__
from .analysis import distribution_summary, median_absolute_deviation, metric_values, trimmed_mean
from .artifacts import file_digest, read_records, write_json


def source_info(path):
    return dict(path=str(path), sha256=file_digest(path))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze saved experimental observations; statistical assumptions apply"
    )
    parser.add_argument(
        "--version", action="version", version="weightscope-analysis " + __version__
    )
    parser.add_argument("command", choices=["distribution"])
    parser.add_argument("--input")
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.command != "paired" and not args.input:
        parser.error(args.command + " requires --input")
    try:
        if args.command == "distribution":
            values = metric_values(read_records(args.input), args.key)
            report = dict(
                source=source_info(args.input),
                metric=args.key,
                distribution=distribution_summary(values),
                mad=median_absolute_deviation(values),
                trimmed=trimmed_mean(values, args.trim_fraction),
            )
        write_json(args.output, report)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
