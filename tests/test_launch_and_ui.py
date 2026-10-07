"""Launchers, local server smoke test, and the confirm-gpu gate on the web API."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.request
from pathlib import Path

from seedregress.cli import main
from seedregress.server import make_server
from tests.mock_comfy import MockComfy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "still-life.suite.json"


def test_icon_and_cover_exist():
    from PIL import Image

    icon = Image.open(ROOT / "assets" / "icon.png")
    assert icon.size == (1024, 1024)
    assert Image.open(ROOT / "assets" / "icon-512.png").size == (512, 512)
    assert (ROOT / "assets" / "icon.svg").read_text(encoding="utf-8").startswith("<svg")
    assert (ROOT / "assets" / "icon.ico").stat().st_size > 100
    icns = (ROOT / "assets" / "icon.icns").read_bytes()
    assert icns[:4] == b"icns"
    cover = Image.open(ROOT / "docs" / "cover.jpg")
    assert cover.size == (1920, 1080)


def test_launcher_files_exist_and_name_the_entry_point():
    command = ROOT / "Launch SeedRegress.command"
    launch = ROOT / "launch.sh"
    bat = ROOT / "Launch SeedRegress.bat"
    desktop = ROOT / "seedregress.desktop"
    for path in (command, launch, bat, desktop):
        assert path.is_file(), path
    assert os.access(command, os.X_OK)
    assert os.access(launch, os.X_OK)
    for path in (command, launch):
        text = path.read_text(encoding="utf-8")
        assert "python -m seedregress serve" in text
        assert "https://www.python.org/downloads/" in text
        assert "3.11" in text
        subprocess.run(["bash", "-n", str(path)], check=True)
    bat_text = bat.read_text(encoding="utf-8")
    assert "python -m seedregress serve" in bat_text
    assert "https://www.python.org/downloads/" in bat_text
    assert "\r" not in command.read_bytes().decode("utf-8")
    desktop_text = desktop.read_text(encoding="utf-8")
    assert "launch.sh" in desktop_text
    assert "SeedRegress" in desktop_text


def test_ui_mentions_confirmation_and_cancel():
    html = (ROOT / "seedregress" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Run on GPU" in html
    assert "Cancel all my jobs" in html
    assert "192.168.4.47:8188" in html
    assert "no scheduler" in html.lower() or "no scheduled" in html.lower() or "There is no" in html


def test_smoke_server_on_loopback(tmp_path):
    httpd = make_server(tmp_path, 0)
    assert httpd.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert "SeedRegress" in body
            assert "Run on GPU" in body
        with urllib.request.urlopen(base + "/assets/icon.png", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("image/png")
        payload = json.dumps({"suite_path": str(EXAMPLE)}).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/plan",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            plan = json.loads(response.read().decode("utf-8"))
        assert plan["mode"] == "dry_run"
        assert plan["queue_checked"] is False
        assert "not contacted" in plan["queue_check"]
        assert len(plan["jobs"]) == 3
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_run_without_confirm_does_not_touch_comfy(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    monkeypatch.setattr("seedregress.comfy._create_connection", boom)
    httpd = make_server(tmp_path, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        payload = json.dumps(
            {"suite_path": str(EXAMPLE), "confirm_gpu": False, "host": "127.0.0.1:9"}
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["mode"] == "dry_run"
        assert body["prompts_submitted"] == 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_cancel_requires_an_explicit_flag(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    httpd = make_server(tmp_path, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        payload = json.dumps({"confirm_cancel": False, "host": "127.0.0.1:9"}).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/cancel",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["contacted"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_plan_is_a_dry_run(capsys, tmp_path):
    code = main(["plan", str(EXAMPLE), "--data-dir", str(tmp_path)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Dry run" in output
    assert "not contacted" in output
    assert "--confirm-gpu" in output


def test_cli_run_without_flag_stays_offline(monkeypatch, capsys, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    monkeypatch.setattr("seedregress.comfy._create_connection", boom)
    code = main(["run", str(EXAMPLE), "--data-dir", str(tmp_path), "--host", "127.0.0.1:9"])
    assert code == 0
    assert "Dry run" in capsys.readouterr().out


def test_build_script_dry_run():
    result = subprocess.run(
        ["python3", str(ROOT / "scripts" / "build_app.py"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PyInstaller" in result.stdout
    assert "icon" in result.stdout.lower()


def test_confirmed_run_through_the_api_uses_the_mock(tmp_path):
    mock = MockComfy()
    host = mock.start()
    httpd = make_server(tmp_path, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        payload = json.dumps(
            {
                "suite_path": str(EXAMPLE),
                "confirm_gpu": True,
                "host": host,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["mode"] == "completed"
        assert body["prompts_submitted"] == 3
        assert body["report_path"]
        with urllib.request.urlopen(base + "/media/" + body["report_path"], timeout=5) as response:
            report = response.read().decode("utf-8")
        assert response.status == 200
        assert "Before" in report and "After" in report and "Diff" in report
    finally:
        httpd.shutdown()
        httpd.server_close()
        mock.stop()
