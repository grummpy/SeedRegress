#!/usr/bin/env python3
"""Optional PyInstaller build.

Run this on the operating system you want a double-clickable build for:

- macOS produces dist/SeedRegress.app (uses assets/icon.icns)
- Windows produces dist/SeedRegress.exe (uses assets/icon.ico)
- Linux produces a dist/SeedRegress folder

Continuous integration does not need to run this. `python scripts/build_app.py --dry-run`
prints the command and exits. A real build needs PyInstaller installed in the
active environment (`pip install pyinstaller`) and is not part of the default launcher.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pyinstaller_command() -> list[str]:
    if sys.platform == "darwin":
        icon = ROOT / "assets" / "icon.icns"
        window = ["--windowed", "--osx-bundle-identifier", "com.seedregress.app"]
    elif sys.platform == "win32":
        icon = ROOT / "assets" / "icon.ico"
        window = ["--console"]
    else:
        icon = ROOT / "assets" / "icon.png"
        window = ["--console"]
    separator = os.pathsep
    data_sets = [
        (ROOT / "assets", "assets"),
        (ROOT / "seedregress" / "static", "seedregress/static"),
        (ROOT / "examples", "examples"),
    ]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        "SeedRegress",
        "--icon",
        str(icon),
        *window,
    ]
    for source, dest in data_sets:
        command.extend(["--add-data", f"{source}{separator}{dest}"])
    command.append(str(ROOT / "seedregress" / "__main__.py"))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the PyInstaller command and do not build",
    )
    args = parser.parse_args(argv)
    command = pyinstaller_command()
    print(" ".join(command))
    if args.dry_run:
        print("Dry run only. No package was built.")
        return 0
    import shutil
    import subprocess

    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("PyInstaller is not installed. pip install pyinstaller", file=sys.stderr)
            return 1
    subprocess.check_call(command, cwd=ROOT)
    print(f"Build finished under {ROOT / 'dist'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
