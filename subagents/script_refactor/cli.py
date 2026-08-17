"""Command-line entry point.

    python -m subagents.script_refactor path/to/flat_script.py
    python -m subagents.script_refactor in.py -o out.py --fine
    python -m subagents.script_refactor in.py --max-statements 15 --summaries

The whole tool is deterministic: naming comes from AST analysis, not a model,
so it needs no credentials and no network and gives the same output every run.
"""

from __future__ import annotations

import argparse
import json
import sys

from .blocking import BlockingConfig
from .refactor import RefactorConfig, refactor_file

#: Single source of truth for the flag defaults, so the CLI cannot drift from
#: the library the way it did when the granularity default changed.
_DEFAULTS = BlockingConfig()


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="script_refactor",
        description="Refactor a flat notebook-derived Python script into functions.",
    )
    parser.add_argument("input", help="flat .py script to refactor")
    parser.add_argument(
        "-o", "--output",
        help="destination file (default: <input>_refactored.py)",
    )
    parser.add_argument(
        "--max-statements",
        type=int,
        default=_DEFAULTS.max_statements,
        help=f"soft ceiling on statements per generated function "
             f"(default: {_DEFAULTS.max_statements})",
    )
    parser.add_argument(
        "--min-statements",
        type=int,
        default=_DEFAULTS.min_statements,
        help=f"never cut a function shorter than this; raise it for fewer, "
             f"larger functions (default: {_DEFAULTS.min_statements})",
    )
    parser.add_argument(
        "--no-constants",
        action="store_true",
        help="do not hoist ALL_CAPS literal assignments to module level",
    )
    # Paired flags over a bare store_true: the default is now ON, and a
    # store_true would pass False whenever the flag was absent, silently
    # overriding that default back to the fine-grained behaviour.
    parser.add_argument(
        "--coarse",
        action="store_true",
        dest="coarse",
        default=_DEFAULTS.keep_chains_together,
        help="fewer, larger functions: an unbroken dependency chain outranks a "
             "change of operation category"
             + (" (default)" if _DEFAULTS.keep_chains_together else ""),
    )
    parser.add_argument(
        "--fine",
        action="store_false",
        dest="coarse",
        help="opposite of --coarse: cut at every change of operation category, "
             "giving many small functions and more generated plumbing",
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="print the per-block summaries as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="analyse and validate, but write nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    config = RefactorConfig(
        blocking=BlockingConfig(
            max_statements=args.max_statements,
            min_statements=args.min_statements,
            keep_chains_together=args.coarse,
        ),
        hoist_constants=not args.no_constants,
    )

    result = refactor_file(
        args.input,
        output_path=None if args.dry_run else args.output,
        config=config,
    ) if not args.dry_run else _dry_run(args, config)

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print(
        f"{len(result.blocks)} function(s): "
        f"{', '.join(block.name for block in result.blocks)}",
        file=sys.stderr,
    )
    if args.summaries:
        print(json.dumps(result.summaries(), indent=2))
    return 0


def _dry_run(args: argparse.Namespace, config: RefactorConfig):
    """Analyse without writing, printing the result to stdout."""
    from pathlib import Path

    from .refactor import refactor_source

    path = Path(args.input)
    config.source_name = path.name
    result = refactor_source(path.read_text(encoding="utf-8"), config)
    if result.ok and not args.summaries:
        print(result.code)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
