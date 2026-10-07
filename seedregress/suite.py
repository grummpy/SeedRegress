"""Load a suite: one ComfyUI API workflow plus the cases to render."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from seedregress.errors import SuiteError

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass
class LoraSpec:
    name: str
    strength_model: float
    strength_clip: float


@dataclass
class Case:
    id: str
    positive: str
    negative: str
    seed: int
    steps: int
    sampler: str
    width: int
    height: int
    loras: list[LoraSpec] = field(default_factory=list)
    scheduler: str | None = None
    cfg: float | None = None
    denoise: float | None = None
    ckpt_name: str | None = None


@dataclass
class Thresholds:
    phash_pass_max: int = 6
    phash_fail_min: int = 16
    ssim_pass_min: float = 0.98
    ssim_fail_max: float = 0.90
    lpips_pass_max: float = 0.08
    lpips_fail_min: float = 0.25

    def to_dict(self) -> dict:
        return {
            "phash_pass_max": self.phash_pass_max,
            "phash_fail_min": self.phash_fail_min,
            "ssim_pass_min": self.ssim_pass_min,
            "ssim_fail_max": self.ssim_fail_max,
            "lpips_pass_max": self.lpips_pass_max,
            "lpips_fail_min": self.lpips_fail_min,
        }


@dataclass
class Bindings:
    positive: str
    negative: str
    sampler: str
    latent: str
    checkpoint: str | None = None
    lora: str | None = None


@dataclass
class Suite:
    name: str
    source_path: Path
    workflow_path: Path
    workflow: dict
    bindings: Bindings
    cases: list[Case]
    thresholds: Thresholds
    seconds_per_step: float | None
    enable_lpips: bool
    comfy_host: str


def load_suite(path: Path) -> Suite:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SuiteError(f"Suite file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SuiteError("A suite file must be a JSON object")

    name = str(raw.get("name") or path.stem).strip()
    if not name:
        raise SuiteError("Suite name is empty")

    workflow_field = raw.get("workflow")
    if not isinstance(workflow_field, str) or not workflow_field.strip():
        raise SuiteError("Suite needs a \"workflow\" path to a ComfyUI API-format JSON file")
    workflow_path = (path.parent / workflow_field).resolve()
    if not workflow_path.is_file():
        raise SuiteError(f"Workflow file not found: {workflow_path}")
    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{workflow_path.name} is not valid JSON: {exc}") from exc
    if not isinstance(workflow, dict) or not workflow:
        raise SuiteError("Workflow JSON must be an object of node id → node")

    bindings = _bindings(raw.get("bindings"), workflow)
    cases = _cases(raw.get("cases"))
    thresholds = _thresholds(raw.get("thresholds") or {})
    seconds = raw.get("seconds_per_step")
    if seconds is not None:
        seconds = float(seconds)
        if seconds <= 0:
            raise SuiteError("seconds_per_step must be greater than zero")
    host = str(raw.get("comfy_host") or "").strip()
    return Suite(
        name=name,
        source_path=path,
        workflow_path=workflow_path,
        workflow=workflow,
        bindings=bindings,
        cases=cases,
        thresholds=thresholds,
        seconds_per_step=seconds,
        enable_lpips=bool(raw.get("enable_lpips", False)),
        comfy_host=host,
    )


def _bindings(raw, workflow: dict) -> Bindings:
    if not isinstance(raw, dict):
        raise SuiteError("Suite needs a \"bindings\" object of node ids")
    required = ("positive", "negative", "sampler", "latent")
    missing = [key for key in required if not str(raw.get(key) or "").strip()]
    if missing:
        raise SuiteError("Suite bindings are missing: " + ", ".join(missing))
    parsed = Bindings(
        positive=str(raw["positive"]),
        negative=str(raw["negative"]),
        sampler=str(raw["sampler"]),
        latent=str(raw["latent"]),
        checkpoint=str(raw["checkpoint"]) if raw.get("checkpoint") else None,
        lora=str(raw["lora"]) if raw.get("lora") else None,
    )
    for label, node_id in (
        ("positive", parsed.positive),
        ("negative", parsed.negative),
        ("sampler", parsed.sampler),
        ("latent", parsed.latent),
        ("checkpoint", parsed.checkpoint),
        ("lora", parsed.lora),
    ):
        if node_id is not None and node_id not in workflow:
            raise SuiteError(f"Binding {label} points at node {node_id}, which is not in the workflow")
    return parsed


def _cases(raw) -> list[Case]:
    if not isinstance(raw, list) or not raw:
        raise SuiteError("Suite needs a non-empty \"cases\" list")
    cases: list[Case] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SuiteError(f"Case {index} must be an object")
        case_id = str(item.get("id") or "").strip()
        if not _CASE_ID.match(case_id):
            raise SuiteError(
                f"Case {index} id must be letters, numbers, '_' or '-' (got {case_id!r})"
            )
        if case_id in seen:
            raise SuiteError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        try:
            width = int(item["width"])
            height = int(item["height"])
            steps = int(item["steps"])
            seed = int(item["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SuiteError(f"Case {case_id} needs integer seed, steps, width, and height") from exc
        if steps < 1:
            raise SuiteError(f"Case {case_id} steps must be at least 1")
        if seed < 0:
            raise SuiteError(f"Case {case_id} seed must be zero or positive")
        if width < 64 or height < 64 or width % 8 or height % 8:
            raise SuiteError(
                f"Case {case_id} width and height must be multiples of 8 and at least 64"
            )
        positive = str(item.get("positive") or "").strip()
        negative = str(item.get("negative") or "").strip()
        sampler = str(item.get("sampler") or "").strip()
        if not positive or not negative or not sampler:
            raise SuiteError(f"Case {case_id} needs positive, negative, and sampler")
        cfg = item.get("cfg")
        denoise = item.get("denoise")
        cases.append(
            Case(
                id=case_id,
                positive=positive,
                negative=negative,
                seed=seed,
                steps=steps,
                sampler=sampler,
                width=width,
                height=height,
                loras=_loras(case_id, item.get("loras") or []),
                scheduler=str(item["scheduler"]) if item.get("scheduler") else None,
                cfg=float(cfg) if cfg is not None else None,
                denoise=float(denoise) if denoise is not None else None,
                ckpt_name=str(item["ckpt_name"]) if item.get("ckpt_name") else None,
            )
        )
    return cases


def _loras(case_id: str, raw) -> list[LoraSpec]:
    if not isinstance(raw, list):
        raise SuiteError(f"Case {case_id} loras must be a list")
    specs: list[LoraSpec] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SuiteError(f"Case {case_id} LoRA {index} must be an object")
        name = str(item.get("name") or "").strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise SuiteError(f"Case {case_id} LoRA {index} needs a file name, not a path")
        model = float(item.get("strength_model", item.get("weight", 1.0)))
        clip = float(item.get("strength_clip", item.get("weight", model)))
        specs.append(LoraSpec(name=name, strength_model=model, strength_clip=clip))
    return specs


def _thresholds(raw) -> Thresholds:
    if not isinstance(raw, dict):
        raise SuiteError("thresholds must be an object")
    base = Thresholds()
    values = base.to_dict()
    for key in values:
        if key in raw and raw[key] is not None:
            values[key] = raw[key]
    parsed = Thresholds(
        phash_pass_max=int(values["phash_pass_max"]),
        phash_fail_min=int(values["phash_fail_min"]),
        ssim_pass_min=float(values["ssim_pass_min"]),
        ssim_fail_max=float(values["ssim_fail_max"]),
        lpips_pass_max=float(values["lpips_pass_max"]),
        lpips_fail_min=float(values["lpips_fail_min"]),
    )
    if parsed.phash_pass_max < 0 or parsed.phash_fail_min <= parsed.phash_pass_max:
        raise SuiteError("pHash thresholds need 0 ≤ pass max < fail min")
    if not 0 <= parsed.ssim_fail_max < parsed.ssim_pass_min <= 1:
        raise SuiteError("SSIM thresholds need 0 ≤ fail max < pass min ≤ 1")
    if parsed.lpips_pass_max < 0 or parsed.lpips_fail_min <= parsed.lpips_pass_max:
        raise SuiteError("LPIPS thresholds need 0 ≤ pass max < fail min")
    return parsed


def case_payload(case: Case) -> dict:
    return {
        "id": case.id,
        "positive": case.positive,
        "negative": case.negative,
        "seed": case.seed,
        "steps": case.steps,
        "sampler": case.sampler,
        "scheduler": case.scheduler,
        "cfg": case.cfg,
        "denoise": case.denoise,
        "width": case.width,
        "height": case.height,
        "ckpt_name": case.ckpt_name,
        "loras": [
            {
                "name": item.name,
                "strength_model": item.strength_model,
                "strength_clip": item.strength_clip,
            }
            for item in case.loras
        ],
    }
