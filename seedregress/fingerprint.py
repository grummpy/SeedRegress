"""Record ComfyUI version and SHA-256 of the checkpoint and LoRA files we can see."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from seedregress.suite import Case, case_payload

_CHECKPOINT_DIRS = (
    "checkpoints",
    "models/checkpoints",
    "diffusion_models",
    "models/diffusion_models",
)
_LORA_DIRS = ("loras", "models/loras")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _find(root: Path, name: str, relatives: tuple[str, ...]) -> tuple[Path | None, str | None]:
    if not root:
        return None, "models_root is not set"
    root = root.expanduser()
    if not root.is_dir():
        return None, "models_root is not a directory"
    candidates = [root / name]
    candidates.extend(root / rel / name for rel in relatives)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if not _inside(root, candidate):
            return None, "path escapes models_root"
        return candidate, None
    return None, "not found under models_root"


def _file_record(root: str, name: str | None, relatives: tuple[str, ...]) -> dict | None:
    if not name:
        return None
    if not root:
        return {"name": name, "sha256": None, "note": "models_root is not set"}
    found, note = _find(Path(root), name, relatives)
    record: dict = {"name": name, "sha256": None, "note": note}
    if found is not None:
        record["sha256"] = sha256_file(found)
        record["note"] = None
        try:
            record["relative_path"] = str(found.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            record["relative_path"] = found.name
    return record


def build_fingerprint(
    *,
    case: Case,
    workflow: dict,
    system_stats: dict | None,
    models_root: str,
) -> dict:
    system = (system_stats or {}).get("system") or {}
    devices = []
    for device in (system_stats or {}).get("devices") or []:
        if isinstance(device, dict):
            devices.append({"name": device.get("name"), "type": device.get("type")})
    workflow_bytes = json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()
    case_bytes = json.dumps(case_payload(case), sort_keys=True, separators=(",", ":")).encode()
    loras = []
    for spec in case.loras:
        record = _file_record(models_root, spec.name, _LORA_DIRS)
        if record is not None:
            record["strength_model"] = spec.strength_model
            record["strength_clip"] = spec.strength_clip
            loras.append(record)
    return {
        "comfyui_version": system.get("comfyui_version"),
        "python_version": system.get("python_version"),
        "pytorch_version": system.get("pytorch_version"),
        "devices": devices,
        "checkpoint": _file_record(models_root, case.ckpt_name, _CHECKPOINT_DIRS),
        "loras": loras,
        "workflow_sha256": sha256_bytes(workflow_bytes),
        "case_sha256": sha256_bytes(case_bytes),
        "case_id": case.id,
    }


def fingerprint_changes(before: dict | None, after: dict) -> list[str]:
    if not before:
        return ["No baseline fingerprint yet."]
    lines: list[str] = []
    if before.get("comfyui_version") != after.get("comfyui_version"):
        lines.append(
            f"ComfyUI version {before.get('comfyui_version')} → {after.get('comfyui_version')}"
        )
    lines.extend(_model_lines("Checkpoint", before.get("checkpoint"), after.get("checkpoint")))
    before_loras = {item.get("name"): item for item in before.get("loras") or []}
    after_loras = {item.get("name"): item for item in after.get("loras") or []}
    for name in sorted(set(before_loras) | set(after_loras)):
        lines.extend(_model_lines(f"LoRA {name}", before_loras.get(name), after_loras.get(name)))
    if before.get("case_sha256") != after.get("case_sha256"):
        lines.append("Case settings (prompt, seed, steps, sampler, size, or LoRA weights) changed.")
    if before.get("workflow_sha256") != after.get("workflow_sha256"):
        lines.append("Workflow JSON changed.")
    return lines or ["No recorded model, LoRA, or ComfyUI version change."]


def _model_lines(label: str, before: dict | None, after: dict | None) -> list[str]:
    if before is None and after is None:
        return []
    if before is None:
        return [f"{label} was added ({after.get('name')})."]
    if after is None:
        return [f"{label} was removed ({before.get('name')})."]
    if before.get("name") != after.get("name"):
        return [f"{label} name {before.get('name')} → {after.get('name')}"]
    if before.get("sha256") != after.get("sha256"):
        return [
            f"{label} file hash { _short(before.get('sha256')) } → { _short(after.get('sha256')) }"
        ]
    return []


def _short(value: str | None) -> str:
    if not value:
        return "unknown"
    return value[:12]
