"""Command line. `run` and `cancel` stay offline unless --confirm-gpu is passed."""

from __future__ import annotations

import argparse
from pathlib import Path

from seedregress import __version__
from seedregress.config import load_config
from seedregress.errors import SeedRegressError
from seedregress.paths import default_data_dir
from seedregress.runner import cancel_mine, execute, exit_code, plan_suite
from seedregress.server import serve
from seedregress.suite import Suite, load_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seedregress",
        description="Visual regression tests for a ComfyUI pipeline. Dry-run is the default.",
    )
    parser.add_argument("--version", action="version", version=f"SeedRegress {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve_parser = sub.add_parser("serve", help="Open the local web UI")
    _add_data(serve_parser)
    serve_parser.add_argument("--port", type=int, default=0, help="Port on 127.0.0.1 (0 picks a free port)")
    serve_parser.add_argument("--no-browser", action="store_true")

    plan_parser = sub.add_parser("plan", help="Show the jobs a run would send. Does not contact ComfyUI.")
    plan_parser.add_argument("suite")
    _add_data(plan_parser)
    plan_parser.add_argument("--host", default="", help="ComfyUI host for this command. Overrides config.")

    run_parser = sub.add_parser("run", help="Dry-run, or render when --confirm-gpu is set")
    run_parser.add_argument("suite")
    _add_data(run_parser)
    run_parser.add_argument("--host", default="")
    run_parser.add_argument("--models-root", default="")
    run_parser.add_argument(
        "--confirm-gpu",
        action="store_true",
        help="Contact ComfyUI and queue jobs. Without this flag, run only prints a dry-run.",
    )
    run_parser.add_argument("--update-baseline", action="store_true")
    run_parser.add_argument("--no-websocket", action="store_true")

    cancel_parser = sub.add_parser(
        "cancel",
        help="Cancel this install's ComfyUI jobs. Other jobs are left in the queue.",
    )
    _add_data(cancel_parser)
    cancel_parser.add_argument("--host", default="")
    cancel_parser.add_argument(
        "--confirm-gpu",
        action="store_true",
        help="Contact ComfyUI. Without this flag, cancel does not open a connection.",
    )

    args = parser.parse_args(argv)
    try:
        if args.cmd == "serve":
            serve(data_dir=Path(args.data_dir), port=args.port, open_browser=not args.no_browser)
            return 0
        if args.cmd == "plan":
            return _print_plan(args, confirm_gpu=False)
        if args.cmd == "run":
            return _print_plan(
                args,
                confirm_gpu=bool(args.confirm_gpu),
                update_baseline=bool(args.update_baseline),
                use_websocket=not args.no_websocket,
            )
        if args.cmd == "cancel":
            return _cancel(args)
    except (SeedRegressError, OSError, ValueError) as exc:
        print(f"SeedRegress: {exc}")
        return 1
    parser.error(f"Unknown command {args.cmd}")
    return 2


def _add_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default="",
        help="Where config, baselines, and reports are stored. Defaults to ./data",
    )


def _data_dir(raw: str) -> Path:
    return Path(raw).expanduser() if raw else default_data_dir()


def _host(args, suite: Suite, config: dict) -> str:
    return (getattr(args, "host", "") or suite.comfy_host or config["comfy_host"]).strip()


def _print_plan(args, *, confirm_gpu: bool, update_baseline: bool = False, use_websocket: bool = True) -> int:
    data_dir = _data_dir(args.data_dir)
    config = load_config(data_dir)
    suite = load_suite(Path(args.suite))
    host = _host(args, suite, config)
    models_root = (getattr(args, "models_root", "") or config["models_root"]).strip()
    if confirm_gpu:
        outcome = execute(
            suite,
            confirm_gpu=True,
            data_dir=data_dir,
            host=host,
            models_root=models_root,
            seconds_per_step=float(config["seconds_per_step"]),
            case_timeout=float(config["case_timeout_seconds"]),
            poll_interval=float(config["poll_interval_seconds"]),
            update_baseline=update_baseline,
            use_websocket=use_websocket,
        )
    else:
        outcome = plan_suite(
            suite,
            data_dir=data_dir,
            host=host,
            seconds_per_step=float(config["seconds_per_step"]),
        )
    _emit(outcome)
    return exit_code(outcome)


def _emit(outcome) -> None:
    print(f"SeedRegress — {outcome.suite}")
    print(f"Mode: {outcome.mode}")
    if outcome.host:
        print(f"Host: {outcome.host}")
    else:
        print("Host: not set")
    print(f"Estimate: {outcome.to_dict()['estimated_label']} ({outcome.to_dict()['estimate_note']})")
    print(f"Queue check: {outcome.queue_check}")
    for warning in outcome.warnings:
        print(f"Warning: {warning}")
    print()
    for job in outcome.jobs:
        loras = ", ".join(item["name"] for item in job.loras) or "none"
        print(
            f"  {job.case_id}: {job.action}, seed {job.seed}, {job.steps} steps, "
            f"{job.sampler}, {job.width}x{job.height}, LoRA {loras}, {job.to_dict()['estimated_label']}"
        )
    if outcome.cases:
        print()
        for case in outcome.cases:
            metrics = []
            if case.phash is not None:
                metrics.append(f"pHash {case.phash}")
            if case.ssim is not None:
                metrics.append(f"SSIM {case.ssim:.4f}")
            extra = f" ({', '.join(metrics)})" if metrics else ""
            print(f"  {case.case_id}: {case.verdict}{extra}")
            for reason in case.reasons:
                print(f"    {reason}")
    if outcome.report_path:
        print(f"\nReport: {outcome.report_path}")
    if outcome.message:
        print(f"\n{outcome.message}")
    if outcome.mode == "dry_run":
        print("Re-run with --confirm-gpu to use the GPU. There is no scheduled auto-run.")


def _cancel(args) -> int:
    data_dir = _data_dir(args.data_dir)
    config = load_config(data_dir)
    host = (args.host or config["comfy_host"]).strip()
    result = cancel_mine(confirm_gpu=bool(args.confirm_gpu), data_dir=data_dir, host=host)
    print(result["message"])
    return 0 if result.get("ok") else 4
