"""CLI entrypoint for iconograph."""

# Pin BLAS / OpenMP thread pools to 1 BEFORE any numpy / cv2 / sklearn import.
# Conditional: only set when this invocation will fork a worker pool. Single-
# process `bench --serial` (or any future `--nproc 1`) skips the cap so the
# master can use full-thread BLAS. The decision must happen pre-import because
# BLAS reads these env vars once on first numpy import; setting them later is
# silently ignored on most builds.
import os
import sys


def _wants_multiproc(argv: list[str]) -> bool:
    """True iff the invocation will fork a process pool. Conservative — only
    `bench --serial` or `--nproc 1` returns False."""
    if "bench" not in argv:
        return True
    if "--serial" in argv:
        return False
    try:
        i = argv.index("--nproc")
    except ValueError:
        return True
    if i + 1 < len(argv) and argv[i + 1] == "1":
        return False
    return True


if _wants_multiproc(sys.argv):
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

import argparse


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

    p_bench = sub.add_parser(
        "bench",
        help="benchmark implemented testbed stages",
    )
    p_bench.add_argument("stage", nargs="?",
                         help="stage name (default: all implemented)")
    p_bench.add_argument("--serial", action="store_true",
                         help="run single-process; skip the BLAS cap")
    p_bench.add_argument("--label", help="tag the .cache/bench JSON file")

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
    p_testbed.add_argument(
        "--panels",
        action="store_true",
        help="decompose: 4-panel qualitative strip [target | D1 | D2 | old L3 beam] per fixture",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "bench":
        from iconograph.testbed import run_bench
        return run_bench(stage=args.stage, serial=args.serial, label=args.label)
    if args.command == "testbed" and args.stage == "categorize":
        from iconograph.testbed import run_categorize_testbed
        return run_categorize_testbed()
    if args.command == "testbed" and args.stage == "preprocess":
        from iconograph.testbed import run_preprocess_testbed
        return run_preprocess_testbed()
    if args.command == "testbed" and args.stage == "align":
        from iconograph.testbed import run_align_testbed
        return run_align_testbed()
    if args.command == "testbed" and args.stage == "segment_prim_classify":
        from iconograph.testbed import run_segment_prim_classify_testbed
        return run_segment_prim_classify_testbed()
    if args.command == "testbed" and args.stage == "L2_feature_segment":
        from iconograph.testbed import run_l2_feature_segment_testbed
        return run_l2_feature_segment_testbed()
    if args.command == "testbed" and args.stage == "decompose":
        from iconograph.testbed import run_decompose_testbed
        return run_decompose_testbed(panels=args.panels)
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
