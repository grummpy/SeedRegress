"""Remember prompt ids this install submitted, so cancel cannot touch other jobs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


class Ownership:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "install.json"
        self.install_id = ""
        self.prompts: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.install_id = str(raw.get("install_id") or "")
            prompts = raw.get("prompts") or {}
            self.prompts = prompts if isinstance(prompts, dict) else {}
        if not self.install_id:
            self.install_id = str(uuid.uuid4())
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"install_id": self.install_id, "prompts": self.prompts}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def remember(self, prompt_id: str, meta: dict) -> None:
        self.prompts[prompt_id] = meta
        self.save()

    def forget(self, prompt_ids: list[str]) -> None:
        for prompt_id in prompt_ids:
            self.prompts.pop(prompt_id, None)
        self.save()

    @property
    def owned_ids(self) -> set[str]:
        return set(self.prompts)
