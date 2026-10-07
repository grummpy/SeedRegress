"""Local config. The ComfyUI host is never given a built-in address."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "comfy_host": "",
    "models_root": "",
    "seconds_per_step": 0.4,
    "case_timeout_seconds": 900,
    "poll_interval_seconds": 0.5,
}


def config_path(data_dir: Path) -> Path:
    return data_dir / "config.json"


def load_config(data_dir: Path) -> dict:
    path = config_path(data_dir)
    loaded = dict(DEFAULT_CONFIG)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be a JSON object")
        for key in DEFAULT_CONFIG:
            if key in raw and raw[key] is not None:
                loaded[key] = raw[key]
    loaded["comfy_host"] = str(loaded["comfy_host"]).strip()
    loaded["models_root"] = str(loaded["models_root"]).strip()
    loaded["seconds_per_step"] = float(loaded["seconds_per_step"])
    loaded["case_timeout_seconds"] = float(loaded["case_timeout_seconds"])
    loaded["poll_interval_seconds"] = float(loaded["poll_interval_seconds"])
    return loaded


def save_config(data_dir: Path, updates: dict) -> dict:
    current = load_config(data_dir)
    for key in DEFAULT_CONFIG:
        if key in updates and updates[key] is not None:
            current[key] = updates[key]
    current["comfy_host"] = str(current["comfy_host"]).strip()
    current["models_root"] = str(current["models_root"]).strip()
    current["seconds_per_step"] = float(current["seconds_per_step"])
    if current["seconds_per_step"] <= 0:
        raise ValueError("seconds_per_step must be greater than zero")
    current["case_timeout_seconds"] = float(current["case_timeout_seconds"])
    current["poll_interval_seconds"] = float(current["poll_interval_seconds"])
    data_dir.mkdir(parents=True, exist_ok=True)
    path = config_path(data_dir)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current
