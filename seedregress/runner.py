"""Plan a run with no network, or render after an explicit GPU confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from seedregress.comfy import (
    ComfyClient,
    item_is_ours,
    parse_queue_item,
    queue_counts,
)
from seedregress.errors import ComfyError, SeedRegressError
from seedregress.fingerprint import build_fingerprint, fingerprint_changes
from seedregress.metrics import abs_diff, compute_lpips, heatmap, lpips_available, phash_distance, ssim
from seedregress.ownership import Ownership
from seedregress.report import write_report
from seedregress.suite import Case, Suite, case_payload
from seedregress.verdict import judge
from seedregress.workflow import apply_case


@dataclass
class PlannedJob:
    case_id: str
    action: str
    seed: int
    steps: int
    sampler: str
    scheduler: str | None
    width: int
    height: int
    loras: list[dict]
    ckpt_name: str | None
    positive: str
    negative: str
    estimated_seconds: float

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "action": self.action,
            "seed": self.seed,
            "steps": self.steps,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "width": self.width,
            "height": self.height,
            "loras": self.loras,
            "ckpt_name": self.ckpt_name,
            "positive": self.positive,
            "negative": self.negative,
            "estimated_seconds": round(self.estimated_seconds, 2),
            "estimated_label": format_duration(self.estimated_seconds),
        }


@dataclass
class CaseOutcome:
    case_id: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    phash: int | None = None
    ssim: float | None = None
    lpips: float | None = None
    max_abs_diff: float | None = None
    size_mismatch: bool = False
    error: str | None = None
    before_path: str | None = None
    after_path: str | None = None
    diff_path: str | None = None
    fingerprint: dict | None = None
    changes: list[str] = field(default_factory=list)
    baseline_updated: bool = False
    submission_unresolved: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "phash": self.phash,
            "ssim": None if self.ssim is None else round(self.ssim, 6),
            "lpips": None if self.lpips is None else round(self.lpips, 6),
            "max_abs_diff": None if self.max_abs_diff is None else round(self.max_abs_diff, 3),
            "size_mismatch": self.size_mismatch,
            "error": self.error,
            "before_path": self.before_path,
            "after_path": self.after_path,
            "diff_path": self.diff_path,
            "fingerprint": self.fingerprint,
            "changes": self.changes,
            "baseline_updated": self.baseline_updated,
            "submission_unresolved": self.submission_unresolved,
        }


@dataclass
class Outcome:
    mode: str
    suite: str
    host: str
    queue_checked: bool
    queue_check: str
    estimated_seconds: float
    jobs: list[PlannedJob]
    prompts_submitted: int = 0
    message: str | None = None
    cases: list[CaseOutcome] = field(default_factory=list)
    report_path: str | None = None
    comfyui_version: str | None = None
    warnings: list[str] = field(default_factory=list)
    running: int = 0
    pending: int = 0

    def to_dict(self, data_dir: Path | None = None) -> dict:
        def rel(path: str | None) -> str | None:
            if path is None or data_dir is None:
                return path
            try:
                return Path(path).resolve().relative_to(data_dir.resolve()).as_posix()
            except ValueError:
                return path

        return {
            "mode": self.mode,
            "suite": self.suite,
            "host": self.host,
            "host_configured": bool(self.host),
            "queue_checked": self.queue_checked,
            "queue_check": self.queue_check,
            "estimated_seconds": round(self.estimated_seconds, 2),
            "estimated_label": format_duration(self.estimated_seconds),
            "estimate_note": (
                "Rough estimate from steps × seconds_per_step, plus a small LoRA allowance. "
                "Not measured on the GPU."
            ),
            "jobs": [job.to_dict() for job in self.jobs],
            "prompts_submitted": self.prompts_submitted,
            "message": self.message,
            "cases": [
                {
                    **case.to_dict(),
                    "before_path": rel(case.before_path),
                    "after_path": rel(case.after_path),
                    "diff_path": rel(case.diff_path),
                }
                for case in self.cases
            ],
            "report_path": rel(self.report_path),
            "comfyui_version": self.comfyui_version,
            "warnings": self.warnings,
            "queue_running": self.running,
            "queue_pending": self.pending,
        }


def format_duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    if whole < 60:
        unit = "second" if whole == 1 else "seconds"
        return f"about {whole} {unit}"
    minutes, sec = divmod(whole, 60)
    if minutes < 60:
        return f"about {minutes} min {sec} sec"
    hours, minutes = divmod(minutes, 60)
    return f"about {hours} hr {minutes} min"


def _estimate(case: Case, seconds_per_step: float) -> float:
    return case.steps * seconds_per_step + 2.0 * len(case.loras)


def _slug(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return cleaned or "suite"


def suite_identity(suite: Suite) -> str:
    """Return the stable on-disk namespace for one suite file.

    A display name is not unique: two suite files may use the same name. The
    source path is stable for an installed suite and keeps their stores apart.
    """
    source = str(suite.source_path.resolve()).encode("utf-8")
    return f"{_slug(suite.name)}--{hashlib.sha256(source).hexdigest()[:12]}"


def baseline_dir(data_dir: Path, suite: Suite, case_id: str) -> Path:
    return data_dir / "baselines" / suite_identity(suite) / case_id


def legacy_baseline_dir(data_dir: Path, suite: Suite, case_id: str) -> Path:
    """Locate the pre-identity baseline path without ever writing to it."""
    return data_dir / "baselines" / _slug(suite.name) / case_id


def plan_suite(
    suite: Suite,
    *,
    data_dir: Path,
    host: str,
    seconds_per_step: float,
) -> Outcome:
    rate = suite.seconds_per_step if suite.seconds_per_step is not None else seconds_per_step
    jobs: list[PlannedJob] = []
    warnings: list[str] = []
    if suite.enable_lpips:
        ready, note = lpips_available()
        if not ready:
            warnings.append(note)
    if not host:
        warnings.append("No ComfyUI host is set. A confirmed run will stop before contacting anything.")
    legacy_cases = [
        case.id for case in suite.cases
        if (legacy_baseline_dir(data_dir, suite, case.id) / "baseline.png").is_file()
        and not (baseline_dir(data_dir, suite, case.id) / "baseline.png").is_file()
    ]
    if legacy_cases:
        warnings.append(
            "Legacy baselines were left untouched because their name-only namespace can belong "
            "to another suite. New isolated baselines will be created for: "
            + ", ".join(legacy_cases)
            + "."
        )
    for case in suite.cases:
        exists = (baseline_dir(data_dir, suite, case.id) / "baseline.png").is_file()
        jobs.append(
            PlannedJob(
                case_id=case.id,
                action="compare to baseline" if exists else "save baseline",
                seed=case.seed,
                steps=case.steps,
                sampler=case.sampler,
                scheduler=case.scheduler,
                width=case.width,
                height=case.height,
                loras=[
                    {
                        "name": item.name,
                        "strength_model": item.strength_model,
                        "strength_clip": item.strength_clip,
                    }
                    for item in case.loras
                ],
                ckpt_name=case.ckpt_name,
                positive=case.positive,
                negative=case.negative,
                estimated_seconds=_estimate(case, rate),
            )
        )
    total = sum(job.estimated_seconds for job in jobs)
    return Outcome(
        mode="dry_run",
        suite=suite.name,
        host=host,
        queue_checked=False,
        queue_check=(
            "not contacted. A confirmed run calls /queue and /system_stats and refuses "
            "if anything is running or pending."
        ),
        estimated_seconds=total,
        jobs=jobs,
        message="Dry run. No jobs were sent.",
        warnings=warnings,
    )


def execute(
    suite: Suite,
    *,
    confirm_gpu: bool,
    data_dir: Path,
    host: str,
    models_root: str = "",
    seconds_per_step: float = 0.4,
    case_timeout: float = 900,
    poll_interval: float = 0.5,
    update_baseline: bool = False,
    use_websocket: bool = True,
) -> Outcome:
    """Dry-run unless ``confirm_gpu`` is exactly True. That path is the only caller of ComfyUI."""
    outcome = plan_suite(
        suite,
        data_dir=data_dir,
        host=host,
        seconds_per_step=seconds_per_step,
    )
    if confirm_gpu is not True:
        return outcome

    if suite.enable_lpips:
        ready, note = lpips_available()
        if not ready:
            outcome.mode = "refused"
            outcome.message = note
            return outcome
    if not host:
        outcome.mode = "refused"
        outcome.message = (
            "Set a ComfyUI host before running. Nothing was contacted."
        )
        return outcome

    data_dir.mkdir(parents=True, exist_ok=True)
    ownership = Ownership(data_dir)
    client = ComfyClient(
        host,
        confirmed=True,
        install_id=ownership.install_id,
        timeout=min(60.0, max(5.0, case_timeout)),
        poll_interval=poll_interval,
        use_websocket=use_websocket,
    )
    try:
        stats = client.system_stats()
        queue = client.get_queue()
    except (ComfyError, SeedRegressError) as exc:
        outcome.mode = "refused"
        outcome.queue_checked = False
        outcome.queue_check = "the queue check failed before any prompt was submitted"
        outcome.message = str(exc)
        return outcome

    outcome.queue_checked = True
    running, pending = queue_counts(queue)
    outcome.running = running
    outcome.pending = pending
    version = (stats.get("system") or {}).get("comfyui_version")
    outcome.comfyui_version = version
    if running or pending:
        outcome.mode = "refused"
        outcome.queue_check = (
            f"busy: {running} running, {pending} pending. Nothing was submitted."
        )
        outcome.message = (
            f"ComfyUI queue is busy ({running} running, {pending} pending). "
            "SeedRegress did not add a job. Cancel this install's jobs first if they are yours, "
            "or wait until the queue is empty."
        )
        return outcome

    outcome.queue_check = "idle (0 running, 0 pending)"
    outcome.mode = "completed"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = data_dir / "runs" / _slug(suite.name) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseOutcome] = []
    for case in suite.cases:
        rendered = _render_case(
                client,
                suite,
                case,
                ownership=ownership,
                data_dir=data_dir,
                run_dir=run_dir,
                run_id=run_id,
                stats=stats,
                models_root=models_root,
                case_timeout=case_timeout,
                update_baseline=update_baseline,
            )
        results.append(rendered)
        if rendered.submission_unresolved:
            results.extend(
                CaseOutcome(
                    case_id=next_case.id,
                    verdict="error",
                    reasons=["Not submitted: a previous submitted render could not be resolved."],
                    error="Not submitted after unresolved prior render.",
                )
                for next_case in suite.cases[len(results):]
            )
            outcome.warnings.append(
                "Stopped submitting new renders after an unresolved submitted job. "
                "No queue cancellation was attempted."
            )
            break
    outcome.prompts_submitted = _count_submitted(run_dir)
    outcome.cases = results
    report = write_report(run_dir, suite.name, results, host=host, comfyui_version=version)
    outcome.report_path = str(report)
    (run_dir / "outcome.json").write_text(
        json.dumps(outcome.to_dict(data_dir), indent=2) + "\n", encoding="utf-8"
    )
    failed = [item.case_id for item in results if item.verdict in {"fail", "error"}]
    drifted = [item.case_id for item in results if item.verdict == "drift"]
    if failed:
        outcome.message = "Finished with failures: " + ", ".join(failed)
    elif drifted:
        outcome.message = "Finished with drift: " + ", ".join(drifted)
    else:
        outcome.message = "Finished."
    return outcome


def _count_submitted(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.glob("*/prompt_id.txt"))


def _render_case(
    client: ComfyClient,
    suite: Suite,
    case: Case,
    *,
    ownership: Ownership,
    data_dir: Path,
    run_dir: Path,
    run_id: str,
    stats: dict,
    models_root: str,
    case_timeout: float,
    update_baseline: bool,
) -> CaseOutcome:
    case_dir = run_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps(case_payload(case), indent=2) + "\n", encoding="utf-8"
    )
    prompt_id: str | None = None
    waiting_for_completion = False
    try:
        graph = apply_case(suite.workflow, case, suite.bindings)
        (case_dir / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        prompt_id = client.submit(graph, run_id=run_id, case_id=case.id)
        (case_dir / "prompt_id.txt").write_text(prompt_id + "\n", encoding="utf-8")
        ownership.remember(
            prompt_id,
            {"run_id": run_id, "case_id": case.id, "host": client.base},
        )
        waiting_for_completion = True
        image_info = client.wait_for_image(prompt_id, case_timeout)
        waiting_for_completion = False
        raw = client.view(image_info["filename"], image_info["subfolder"], image_info["type"])
        current = Image.open(BytesIO(raw)).convert("RGB")
        after_path = case_dir / "after.png"
        current.save(after_path)
        ownership.forget([prompt_id])
    except (ComfyError, SeedRegressError, OSError, ValueError) as exc:
        unresolved = prompt_id is not None and waiting_for_completion
        reason = str(exc)
        if unresolved:
            reason += " The submitted job may still be queued or running; no cancellation was attempted."
        return CaseOutcome(
            case_id=case.id,
            verdict="error",
            error=reason,
            reasons=[reason],
            submission_unresolved=unresolved,
        )

    fingerprint = build_fingerprint(
        case=case, workflow=graph, system_stats=stats, models_root=models_root
    )
    (case_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
    )
    base = baseline_dir(data_dir, suite, case.id)
    base.mkdir(parents=True, exist_ok=True)
    metadata_path = base / "suite-identity.json"
    if not metadata_path.is_file():
        metadata_path.write_text(
            json.dumps(
                {
                    "identity": suite_identity(suite),
                    "name": suite.name,
                    "source_path": str(suite.source_path.resolve()),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    baseline_png = base / "baseline.png"
    baseline_fp_path = base / "fingerprint.json"
    if not baseline_png.is_file():
        current.save(baseline_png)
        baseline_fp_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
        return CaseOutcome(
            case_id=case.id,
            verdict="baseline",
            reasons=["No baseline existed. This render was saved and not scored."],
            after_path=str(after_path),
            before_path=str(baseline_png),
            fingerprint=fingerprint,
            changes=["Baseline created."],
        )

    before = Image.open(baseline_png).convert("RGB")
    before_copy = case_dir / "before.png"
    before.save(before_copy)
    previous = None
    if baseline_fp_path.is_file():
        previous = json.loads(baseline_fp_path.read_text(encoding="utf-8"))
    mismatch = before.size != current.size
    diff = heatmap(before, current)
    diff_path = case_dir / "diff.png"
    diff.save(diff_path)
    distance = phash_distance(before, current)
    similarity = ssim(before, current)
    lpips_value = None
    if suite.enable_lpips:
        lpips_value = compute_lpips(before, current)
    verdict, reasons = judge(
        phash=distance,
        ssim_value=similarity,
        thresholds=suite.thresholds,
        lpips_value=lpips_value,
        lpips_enabled=suite.enable_lpips,
        size_mismatch=mismatch,
    )
    if update_baseline:
        current.save(baseline_png)
        baseline_fp_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    return CaseOutcome(
        case_id=case.id,
        verdict=verdict,
        reasons=reasons,
        phash=distance,
        ssim=similarity,
        lpips=lpips_value,
        max_abs_diff=float(abs_diff(before, current).max()),
        size_mismatch=mismatch,
        before_path=str(before_copy),
        after_path=str(after_path),
        diff_path=str(diff_path),
        fingerprint=fingerprint,
        changes=fingerprint_changes(previous, fingerprint),
        baseline_updated=update_baseline,
    )


def cancel_mine(
    *,
    confirm_gpu: bool,
    data_dir: Path,
    host: str,
) -> dict:
    """Interrupt and delete only this install's jobs. Requires the same confirmation flag."""
    if confirm_gpu is not True:
        return {
            "ok": False,
            "contacted": False,
            "message": "Pass --confirm-gpu to contact ComfyUI. Nothing was cancelled.",
        }
    if not host:
        return {
            "ok": False,
            "contacted": False,
            "message": "Set a ComfyUI host first. Nothing was contacted.",
        }
    ownership = Ownership(data_dir)
    client = ComfyClient(host, confirmed=True, install_id=ownership.install_id, use_websocket=False)
    queue = client.get_queue()
    to_delete: list[str] = []
    interrupt = False
    spared: list[str] = []
    for item in queue.get("queue_pending") or []:
        prompt_id, extra = parse_queue_item(item)
        if item_is_ours(prompt_id, extra, ownership.owned_ids, ownership.install_id):
            if prompt_id:
                to_delete.append(prompt_id)
        elif prompt_id:
            spared.append(prompt_id)
    for item in queue.get("queue_running") or []:
        prompt_id, extra = parse_queue_item(item)
        if item_is_ours(prompt_id, extra, ownership.owned_ids, ownership.install_id):
            interrupt = True
        elif prompt_id:
            spared.append(prompt_id)
    if to_delete:
        client.delete_queue_items(to_delete)
        ownership.forget(to_delete)
    if interrupt:
        client.interrupt()
    return {
        "ok": True,
        "contacted": True,
        "deleted": to_delete,
        "interrupted": interrupt,
        "spared": spared,
        "message": (
            f"Deleted {len(to_delete)} pending job(s) from this install. "
            + ("Interrupted the running job." if interrupt else "Did not interrupt the running job.")
            + (f" Left {len(spared)} other job(s) in place." if spared else "")
        ),
    }


def exit_code(outcome: Outcome) -> int:
    if outcome.mode == "dry_run":
        return 0
    if outcome.mode == "refused":
        return 4
    if any(item.verdict in {"fail", "error"} for item in outcome.cases):
        return 3
    if any(item.verdict == "drift" for item in outcome.cases):
        return 2
    return 0
