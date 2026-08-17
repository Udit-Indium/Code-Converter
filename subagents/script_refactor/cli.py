"""Command-line entry point.

    python -m subagents.script_refactor path/to/flat_script.py
    python -m subagents.script_refactor in.py -o out.py --llm
    python -m subagents.script_refactor in.py --max-statements 15 --summaries

Deterministic naming is the default: it needs no credentials and no network, so
the tool is useful — and testable — with nothing configured. `--llm` opts into
model-generated names.
"""

from __future__ import annotations

import argparse
import json
import sys

from .blocking import BlockingConfig
from .naming import LLMFunctionNamer
from .refactor import RefactorConfig, refactor_file


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
        "--llm",
        action="store_true",
        help="use an LLM to name functions (summaries only; needs credentials)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="LiteLLM model id for --llm (default: the repo's REFACTOR_MODEL)",
    )
    parser.add_argument(
        "--max-statements",
        type=int,
        default=25,
        help="soft ceiling on statements per generated function (default: 25)",
    )
    parser.add_argument(
        "--no-constants",
        action="store_true",
        help="do not hoist ALL_CAPS literal assignments to module level",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="fewer, larger functions: an unbroken dependency chain outranks a "
             "change of operation category",
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


def _make_namer(args: argparse.Namespace) -> LLMFunctionNamer | None:
    """Build the LLM namer, if one was asked for."""
    if not args.llm:
        return None
    model = args.model
    if not model:
        # Imported lazily so the CLI works standalone, outside the agent repo.
        from ..model_config import REFACTOR_MODEL

        model = f"databricks/{REFACTOR_MODEL}"
    return LLMFunctionNamer(model=model)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    config = RefactorConfig(
        blocking=BlockingConfig(
            max_statements=args.max_statements,
            keep_chains_together=args.coarse,
        ),
        namer=_make_namer(args),
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
