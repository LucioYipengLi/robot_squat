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
        "--planar-sampling",
        choices=("legacy", "transition"),
        help="Evaluate pure planar targets with a shared sampling mode instead of the checkpoint setting",
    )
    mine.add_argument(
        "--planar-near-lower-scale",
        type=float,
        metavar="SCALE",
        help=(
            "One planar episode per environment. After first near entry, cap lateral/yaw to SCALE of their ranges. "
            "Use 1 for the baseline."
        ),
    )
    mine.add_argument(
        "--planar-startup",
        action="store_true",
        help="Run the first episode per environment and save the first 0.2 s of every wide episode",
    )
    hold = mine.add_mutually_exclusive_group()
    hold.add_argument(
        "--planar-hold-from",
        type=Path,
        metavar="BASELINE",
        help="Hold both arms during startup and replay Lower/waist commands from a --planar-startup archive",
    )
    hold.add_argument(
        "--planar-hold-waist-from",
        type=Path,
        metavar="ARM_HOLD_RUN",
        help="Hold waist and arms during startup, replay the original Lower inputs and compare to an arms-only run",
    )
    mine.add_argument(
        "--regular_refresh_s",
        type=positive_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Override regular goal hold bounds in evaluation (default: checkpoint bounds)",
    )
    mine.add_argument(
        "--regular-sampling",
        choices=("checkpoint", "native"),
        help="Restore training sampling or use unbiased regular refreshes (default: checkpoint)",
    )
    mine.add_argument("--policy_output", choices=("mean", "sample"), default="mean")
    mine.add_argument(
        "--switch_pairs", type=Path, help="Run fixed regular switches as single-arm mirrors or dual-arm cases"
    )
    mine.add_argument(
        "--switch-diagnostic", choices=("direct", "hold", "retreat"), help="Compare fixed switches with a Lower retreat"
    )
    mine.add_argument("--switch-cases", nargs="+", metavar="ID", help="Run only these case IDs from the fixed suite")
    mine.add_argument("--seed", type=int, default=42)
    mine.add_argument("--dr-phase", type=int, choices=(1, 2, 3, 4), default=1, help="Evaluate with this DR phase")
    mine.add_argument("--dr-push-delta-v", type=positive_float, help="Override phase 4 push bounds per axis (m/s)")
    mine.add_argument("--device", default="cuda:0")
    view = modes.add_parser("view", help="Serve a local 3-D replay/force timeline; no simulator required")
    view.add_argument("--run", type=Path, required=True)
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no_browser", action="store_true")
    pairs = modes.add_parser("pairs", help="Build a fixed switch-pair suite from an existing failure archive")
    pairs.add_argument("--run", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True, help="New suite JSON path")
    pairs.add_argument("--suite", choices=("inward", "retract", "dual_retract"), default="inward")
    pairs.add_argument(
        "--per-bin", type=positive_int, default=50, help="Cases per jump-distance bin in a retract suite"
    )
    pairs.add_argument("--in_up", type=positive_int, default=100)
    pairs.add_argument("--controls", type=positive_int, default=20, help="Pairs per control group")
    pairs.add_argument("--regressions", type=int, default=10, help="Maximum additional known failure pairs")
    pairs.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.mode == "mine" and args.level is not None and args.level < 0:
        parser.error("--level must be non-negative")
    if args.mode == "mine" and args.dr_push_delta_v is not None and args.dr_phase != 4:
        parser.error("--dr-push-delta-v requires --dr-phase 4")
    if args.mode == "mine" and args.planar_sampling is not None and args.category != "planar":
        parser.error("--planar-sampling requires --category planar")
    if args.mode == "mine" and args.planar_startup and args.planar_sampling != "transition":
        parser.error("--planar-startup requires --planar-sampling transition")
    if (
        args.mode == "mine"
        and (args.planar_hold_from is not None or args.planar_hold_waist_from is not None)
        and (
            args.planar_sampling != "transition"
            or args.policy_output != "mean"
            or args.planar_near_lower_scale is not None
        )
    ):
        parser.error("Planar hold requires --planar-sampling transition, mean output and no Lower cap")
    if args.mode == "mine" and args.planar_near_lower_scale is not None:
        if not 0.0 <= args.planar_near_lower_scale <= 1.0:
            parser.error("--planar-near-lower-scale must be between 0 and 1")
        if args.planar_sampling is None or args.policy_output != "mean":
            parser.error("--planar-near-lower-scale requires explicit --planar-sampling and mean output")
    if (
        args.mode == "mine"
        and args.switch_pairs is not None
        and (args.category != "regular" or args.policy_output != "mean" or args.regular_refresh_s is not None)
    ):
        parser.error("--switch_pairs requires --category regular and mean output, with no --regular_refresh_s")
    if args.mode == "mine" and args.switch_pairs is None and (args.switch_diagnostic or args.switch_cases):
        parser.error("--switch-diagnostic and --switch-cases require --switch_pairs")
    if args.mode == "mine" and args.regular_refresh_s is not None:
        if args.category not in ("regular", "mixed"):
            parser.error("--regular_refresh_s requires --category regular or mixed")
        if args.regular_refresh_s[0] > args.regular_refresh_s[1]:
            parser.error("--regular_refresh_s requires MIN <= MAX")
    if (
        args.mode == "mine"
        and args.regular_sampling is not None
        and (args.category not in ("regular", "mixed") or args.switch_pairs is not None)
    ):
        parser.error("--regular-sampling requires --category regular or mixed and ordinary goal refreshes")
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
            suite=args.suite,
            per_bin=args.per_bin,
        )
    from .viewer import serve

    return serve(args.run, port=args.port, open_browser=not args.no_browser)
