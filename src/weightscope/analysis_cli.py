"""Command-line research diagnostics for saved experiment observations."""

import argparse
import json

from . import __version__
from .analysis import (
    align_pairs,
    distribution_summary,
    group_summaries,
    median_absolute_deviation,
    metric_values,
    paired_bootstrap,
    paired_permutation,
    paired_summary,
    trimmed_mean,
)
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
    parser.add_argument("command", choices=["distribution", "paired", "grouped"])
    parser.add_argument("--input")
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--id-key", default="seed")
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-key")
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
        elif args.command == "paired":
            if not args.before or not args.after:
                parser.error("paired requires --before and --after")
            pairs = align_pairs(
                read_records(args.before),
                read_records(args.after),
                args.id_key,
                args.key,
                not args.allow_unmatched,
            )
            report = dict(
                before_source=source_info(args.before),
                after_source=source_info(args.after),
                metric=args.key,
                pairs=pairs,
                descriptive=paired_summary(pairs["before"], pairs["after"]),
                bootstrap=paired_bootstrap(
                    pairs["before"], pairs["after"], args.samples, args.confidence, args.seed
                ),
                permutation=paired_permutation(
                    pairs["before"], pairs["after"], args.samples, args.seed
                ),
            )
        elif args.command == "grouped":
            if not args.group_key:
                parser.error("grouped requires --group-key")
            report = dict(
                source=source_info(args.input),
                metric=args.key,
                group_key=args.group_key,
                groups=group_summaries(read_records(args.input), args.group_key, args.key),
            )
        write_json(args.output, report)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
