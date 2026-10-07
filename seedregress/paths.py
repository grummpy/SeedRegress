"""Locations for source assets and the gitignored data directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def package_root() -> Path:
    """Repo root in a checkout, or the PyInstaller bundle root when frozen."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    override = os.environ.get("SEEDREGRESS_DATA", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "data"
