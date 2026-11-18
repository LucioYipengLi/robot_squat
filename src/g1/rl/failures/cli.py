"""Simulator-free argument parsing; the viewer also works without Isaac Sim or a GPU."""

import argparse
import math
from pathlib import Path


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Mine Upper Mobile failures and inspect recorded motion/contact data.")
    modes = parser.add_subparsers(dest="mode", required=True)
    mine = modes.add_parser("mine", help="Batch inference with no window, rendering or optimizer updates")
    mine.add_argument("--task", choices=("Template-G1-Upper-Mobile-v0",), default="Template-G1-Upper-Mobile-v0")
    mine.add_argument("--checkpoint", type=Path, required=True)
    mine.add_argument(
        "--level", type=int, help="Freeze collection at this curriculum level (default: checkpoint level)"
    )
    mine.add_argument("--output", type=Path, help="New output directory (default: outputs/failures/<timestamp>)")
    mine.add_argument("--num_envs", type=positive_int, default=256)
    mine.add_argument("--episodes", type=positive_int, default=5000, help="Maximum completed episodes")
    mine.add_argument("--max_failures", type=positive_int, default=100, help="Stop after saving this many failures")
    mine.add_argument("--max_steps", type=positive_int, default=30000, help="Maximum vectorized control steps")
    mine.add_argument(
        "--history_s", type=positive_float, default=2.0, help="Motion/contact history before each failure"
    )
    mine.add_argument("--category", choices=("planar", "regular", "height_shifted", "mixed"), default="planar")
    mine.add_argument(
        "--regular_refresh_s",
        type=positive_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override regular goal hold bounds in evaluation (default: checkpoint bounds)",
    )
    mine.add_argument("--policy_output", choices=("mean", "sample"), default="mean")
    mine.add_argument("--switch_pairs", type=Path, help="Run each fixed regular switch pair once per side")
    mine.add_argument("--seed", type=int, default=42)
    mine.add_argument("--device", default="cuda:0")
    view = modes.add_parser("view", help="Serve a local 3-D replay/force timeline; no simulator required")
    view.add_argument("--run", type=Path, required=True)
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no_browser", action="store_true")
    pairs = modes.add_parser("pairs", help="Build a fixed switch-pair suite from an existing failure archive")
    pairs.add_argument("--run", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True, help="New suite JSON path")
    pairs.add_argument("--in_up", type=positive_int, default=100)
    pairs.add_argument("--controls", type=positive_int, default=20, help="Pairs per control group")
    pairs.add_argument("--regressions", type=int, default=10, help="Maximum additional known failure pairs")
    pairs.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.mode == "mine" and args.level is not None and args.level < 0:
        parser.error("--level must be non-negative")
    if (
        args.mode == "mine"
        and args.switch_pairs is not None
        and (args.category != "regular" or args.policy_output != "mean" or args.regular_refresh_s is not None)
    ):
        parser.error("--switch_pairs requires --category regular and mean output, with no --regular_refresh_s")
    if args.mode == "mine" and args.regular_refresh_s is not None:
        if args.category not in ("regular", "mixed"):
            parser.error("--regular_refresh_s requires --category regular or mixed")
        if args.regular_refresh_s[0] > args.regular_refresh_s[1]:
            parser.error("--regular_refresh_s requires MIN <= MAX")
    if args.mode == "view" and not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.mode == "pairs" and args.regressions < 0:
        parser.error("--regressions must be non-negative")
    return args


def run(argv):
    args = parse_args(argv)
    if args.mode == "mine":
        from .mine import run_mining

        return run_mining(args)
    if args.mode == "pairs":
        from .pairs import generate_switch_pairs

        return generate_switch_pairs(
            args.run,
            args.output,
            in_up=args.in_up,
            controls=args.controls,
            regressions=args.regressions,
            seed=args.seed,
        )
    from .viewer import serve

    return serve(args.run, port=args.port, open_browser=not args.no_browser)
