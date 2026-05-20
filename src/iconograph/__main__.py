"""CLI entrypoint for iconograph."""

import argparse
import sys


def _not_implemented(name: str) -> int:
    print(f"iconograph: {name} not implemented", file=sys.stderr)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iconograph")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build",
        help="batch over inputs/logo-markings/, skip unchanged",
    )
    p_build.add_argument("--force", action="store_true", help="reprocess everything")
    p_build.add_argument("--only", metavar="GLOB", help="reprocess matching inputs")

    sub.add_parser("review", help="regenerate review.html")

    p_testbed = sub.add_parser(
        "testbed",
        help="run an in-development stage against cached previous-stage outputs",
    )
    p_testbed.add_argument("stage", nargs="?")
    p_testbed.add_argument("branch", nargs="?")
    p_testbed.add_argument(
        "--compare",
        nargs="+",
        metavar="VARIANT",
        help="render multiple proposed methodologies side-by-side",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "testbed" and args.stage == "categorize":
        from iconograph.testbed import run_categorize_testbed
        return run_categorize_testbed()
    if args.command == "testbed" and args.stage == "preprocess":
        from iconograph.testbed import run_preprocess_testbed
        return run_preprocess_testbed()
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
