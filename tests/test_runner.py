"""GPU confirmation, busy-queue refusal, mock ComfyUI, cancel, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedregress.comfy import ComfyClient, normalize_base, read_frame, send_frame
from seedregress.config import load_config
from seedregress.errors import AmbiguousSubmission, ComfyError, GpuConfirmationRequired, TerminalPromptFailure
from seedregress.runner import baseline_dir, cancel_mine, execute, exit_code, plan_suite
from seedregress.suite import LoraSpec, load_suite
from seedregress.workflow import apply_case
from tests.mock_comfy import MockComfy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "still-life.suite.json"


@pytest.fixture
def suite():
    return load_suite(EXAMPLE)


def test_example_suite_is_safe_and_has_no_baked_in_host(suite):
    assert suite.comfy_host == ""
    assert suite.enable_lpips is False
    blob = EXAMPLE.read_text(encoding="utf-8").lower()
    for banned in ("nude", "nsfw", "child", "gore", "blood"):
        assert banned not in blob
    positives = " ".join(case.positive for case in suite.cases)
    assert "mountain" in positives
    assert "ceramic bowl" in positives
    assert "geometric solids" in positives


def test_workflow_patches_prompts_and_loras(suite):
    ridge = suite.cases[0]
    graph = apply_case(suite.workflow, ridge, suite.bindings)
    assert graph["6"]["inputs"]["text"] == ridge.positive
    assert graph["7"]["inputs"]["text"] == ridge.negative
    assert graph["3"]["inputs"]["seed"] == ridge.seed
    assert graph["3"]["inputs"]["steps"] == ridge.steps
    assert graph["3"]["inputs"]["sampler_name"] == ridge.sampler
    assert graph["5"]["inputs"]["width"] == 512
    assert graph["10"]["inputs"]["lora_name"] == "landscape_detail.safetensors"

    bowl = suite.cases[1]
    bypassed = apply_case(suite.workflow, bowl, suite.bindings)
    assert "10" not in bypassed
    assert bypassed["3"]["inputs"]["model"] == ["4", 0]
    assert bypassed["6"]["inputs"]["clip"] == ["4", 1]
    assert bypassed["5"]["inputs"]["width"] == 768

    doubled = apply_case(suite.workflow, ridge, suite.bindings)
    ridge.loras.append(LoraSpec("shape_clean.safetensors", 0.2, 0.3))
    chained = apply_case(suite.workflow, ridge, suite.bindings)
    assert chained["10"]["inputs"]["lora_name"] == "landscape_detail.safetensors"
    assert chained["10_sr1"]["inputs"]["lora_name"] == "shape_clean.safetensors"
    assert chained["10_sr1"]["inputs"]["model"] == ["10", 0]
    assert chained["3"]["inputs"]["model"] == ["10_sr1", 0]
    assert doubled["3"]["inputs"]["model"] == ["10", 0]


def test_dry_run_makes_no_network_call(monkeypatch, suite, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    monkeypatch.setattr("seedregress.comfy._create_connection", boom)
    outcome = execute(suite, confirm_gpu=False, data_dir=tmp_path, host="127.0.0.1:9")
    assert outcome.mode == "dry_run"
    assert outcome.queue_checked is False
    assert outcome.prompts_submitted == 0
    assert "not contacted" in outcome.queue_check
    assert len(outcome.jobs) == 3
    assert exit_code(outcome) == 0


def test_client_refuses_to_exist_without_confirmation():
    with pytest.raises(GpuConfirmationRequired):
        ComfyClient("127.0.0.1:9", confirmed=False, install_id="x")


def test_empty_host_refuses_before_any_socket(monkeypatch, suite, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    monkeypatch.setattr("seedregress.comfy._create_connection", boom)
    outcome = execute(suite, confirm_gpu=True, data_dir=tmp_path, host="")
    assert outcome.mode == "refused"
    assert outcome.prompts_submitted == 0
    assert "Nothing was contacted" in outcome.message


def test_lpips_requested_but_missing_does_not_call_out(monkeypatch, suite, tmp_path):
    suite.enable_lpips = True

    def boom(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("seedregress.comfy._urlopen", boom)
    monkeypatch.setattr("seedregress.metrics.lpips_available", lambda: (False, "LPIPS missing"))
    outcome = execute(suite, confirm_gpu=True, data_dir=tmp_path, host="127.0.0.1:9")
    assert outcome.mode == "refused"
    assert "LPIPS" in outcome.message


def test_busy_queue_submits_nothing(suite, tmp_path):
    mock = MockComfy()
    host = mock.start()
    mock.state.queue_pending.append([1, "someone-else", {}, {"client_id": "other"}, []])
    try:
        outcome = execute(
            suite,
            confirm_gpu=True,
            data_dir=tmp_path,
            host=host,
            poll_interval=0.01,
            case_timeout=5,
        )
    finally:
        mock.stop()
    assert outcome.mode == "refused"
    assert outcome.queue_checked is True
    assert outcome.prompts_submitted == 0
    assert outcome.pending == 1
    assert mock.prompt_posts() == []
    assert ("GET", "/queue") in mock.state.calls
    assert ("GET", "/system_stats") in mock.state.calls
    assert mock.state.cleared is False


def test_mock_roundtrip_baseline_then_pass_and_hashes(suite, tmp_path):
    models = tmp_path / "models"
    (models / "checkpoints").mkdir(parents=True)
    (models / "loras").mkdir()
    checkpoint = models / "checkpoints" / "v1-5-pruned-emaonly.safetensors"
    lora = models / "loras" / "landscape_detail.safetensors"
    checkpoint.write_bytes(b"checkpoint-bytes")
    lora.write_bytes(b"lora-bytes")
    mock = MockComfy()
    host = mock.start()
    try:
        first = execute(
            suite,
            confirm_gpu=True,
            data_dir=tmp_path,
            host=host,
            models_root=str(models),
            poll_interval=0.01,
            case_timeout=5,
            use_websocket=True,
        )
        assert first.mode == "completed"
        assert first.prompts_submitted == 3
        assert first.comfyui_version == "0.3.43"
        assert {item.verdict for item in first.cases} == {"baseline"}
        assert ("POST", "/prompt") in mock.state.calls
        assert any(path == "/history/prompt-1" for _method, path in mock.state.calls if path.startswith("/history"))
        assert ("GET", "/view") in mock.state.calls
        posted = mock.prompt_posts()
        bowl = next(body for body in posted if body["extra_data"]["seedregress_case"] == "ceramic-bowl")
        assert "10" not in bowl["prompt"]
        ridge = next(body for body in posted if body["extra_data"]["seedregress_case"] == "ridge-dusk")
        assert ridge["prompt"]["10"]["inputs"]["lora_name"] == "landscape_detail.safetensors"
        fingerprint = json.loads(
            (baseline_dir(tmp_path, suite, "ridge-dusk") / "fingerprint.json").read_text()
        )
        assert fingerprint["comfyui_version"] == "0.3.43"
        assert fingerprint["checkpoint"]["sha256"]
        assert fingerprint["loras"][0]["sha256"]
        assert "argv" not in json.dumps(fingerprint)

        second = execute(
            suite,
            confirm_gpu=True,
            data_dir=tmp_path,
            host=host,
            models_root=str(models),
            poll_interval=0.01,
            case_timeout=5,
        )
        assert {item.verdict for item in second.cases} == {"pass"}
        assert second.report_path
        report = Path(second.report_path).read_text(encoding="utf-8")
        assert "Before" in report and "After" in report and "Diff" in report
        assert "pass" in report
        mock.state.salt = 9
        third = execute(
            suite,
            confirm_gpu=True,
            data_dir=tmp_path,
            host=host,
            models_root=str(models),
            poll_interval=0.01,
            case_timeout=5,
        )
        assert {item.verdict for item in third.cases} == {"fail"}
        assert exit_code(third) == 3
    finally:
        mock.stop()


def test_cancel_touches_only_this_install(tmp_path):
    mock = MockComfy()
    host = mock.start()
    try:
        from seedregress.ownership import Ownership

        ownership = Ownership(tmp_path)
        ownership.remember("ours-pending", {"case_id": "ridge-dusk"})
        mock.state.queue_running.append(
            [0, "ours-running", {}, {"seedregress_install": ownership.install_id}, []]
        )
        mock.state.queue_pending.append([1, "ours-pending", {}, {}, []])
        mock.state.queue_pending.append([2, "foreign", {}, {"client_id": "other-app"}, []])
        refused = cancel_mine(confirm_gpu=False, data_dir=tmp_path, host=host)
        assert refused["contacted"] is False
        assert mock.state.calls == []
        result = cancel_mine(confirm_gpu=True, data_dir=tmp_path, host=host)
        assert result["deleted"] == ["ours-pending"]
        assert result["interrupted"] is True
        assert result["spared"] == ["foreign"]
        assert mock.state.cleared is False
        delete_bodies = [
            body for method, path, body in mock.state.bodies if path == "/queue" and method == "POST"
        ]
        assert delete_bodies == [{"delete": ["ours-pending"]}]
        assert ("POST", "/interrupt") in mock.state.calls
    finally:
        mock.stop()


def test_plan_describes_baseline_versus_compare(suite, tmp_path):
    outcome = plan_suite(suite, data_dir=tmp_path, host="", seconds_per_step=0.4)
    assert all(job.action == "save baseline" for job in outcome.jobs)
    assert "about" in outcome.to_dict()["estimated_label"]


def test_unresolved_submission_stops_later_prompts_and_a_retry_can_continue(monkeypatch, suite, tmp_path):
    mock = MockComfy()
    host = mock.start()
    original_wait = ComfyClient.wait_for_image
    calls = 0

    def fail_once(self, prompt_id, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ComfyError("timed out waiting for mock completion")
        return original_wait(self, prompt_id, timeout)

    monkeypatch.setattr(ComfyClient, "wait_for_image", fail_once)
    try:
        failed = execute(
            suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5
        )
        assert failed.prompts_submitted == 1
        assert [item.case_id for item in failed.cases] == [case.id for case in suite.cases]
        assert failed.cases[0].submission_unresolved is True
        assert all(item.verdict == "error" for item in failed.cases)
        assert len(mock.prompt_posts()) == 1
        assert mock.state.interrupted == 0

        retried = execute(
            suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5
        )
        assert retried.prompts_submitted == len(suite.cases)
        assert {item.verdict for item in retried.cases} == {"baseline"}
    finally:
        mock.stop()


def test_ambiguous_pre_id_submission_stops_later_prompts(monkeypatch, suite, tmp_path):
    mock = MockComfy()
    host = mock.start()

    def ambiguous_submit(self, graph, *, run_id, case_id):
        raise AmbiguousSubmission("response lost before prompt id")

    monkeypatch.setattr(ComfyClient, "submit", ambiguous_submit)
    try:
        outcome = execute(suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5)
    finally:
        mock.stop()
    assert outcome.prompts_submitted == 0
    assert outcome.cases[0].submission_unresolved is True
    assert [item.case_id for item in outcome.cases] == [case.id for case in suite.cases]
    assert all(item.verdict == "error" for item in outcome.cases)
    assert mock.prompt_posts() == []


def test_terminal_prompt_failure_does_not_stop_independent_cases(monkeypatch, suite, tmp_path):
    mock = MockComfy()
    host = mock.start()
    original_wait = ComfyClient.wait_for_image
    calls = 0

    def terminal_once(self, prompt_id, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TerminalPromptFailure("validation failed")
        return original_wait(self, prompt_id, timeout)

    monkeypatch.setattr(ComfyClient, "wait_for_image", terminal_once)
    try:
        outcome = execute(suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5)
    finally:
        mock.stop()
    assert outcome.prompts_submitted == len(suite.cases)
    assert outcome.cases[0].submission_unresolved is False
    assert [item.verdict for item in outcome.cases] == ["error", "baseline", "baseline"]


def test_reports_created_in_same_second_do_not_share_a_directory(monkeypatch, suite, tmp_path):
    mock = MockComfy()
    host = mock.start()
    try:
        first = execute(suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5)
        second = execute(suite, confirm_gpu=True, data_dir=tmp_path, host=host, poll_interval=0.01, case_timeout=5)
    finally:
        mock.stop()
    assert Path(first.report_path).parents[1] != Path(second.report_path).parents[1]


def test_same_name_suites_do_not_share_baselines_and_legacy_is_not_adopted(suite, tmp_path):
    from copy import copy

    other_path = tmp_path / "other" / "same-name.suite.json"
    other_path.parent.mkdir()
    other_path.write_text("{}", encoding="utf-8")
    other = copy(suite)
    other.source_path = other_path
    assert baseline_dir(tmp_path, suite, "ridge-dusk") != baseline_dir(tmp_path, other, "ridge-dusk")

    legacy = tmp_path / "baselines" / "still-landscapes-and-objects" / "ridge-dusk"
    legacy.mkdir(parents=True)
    (legacy / "baseline.png").write_bytes(b"legacy")
    plan = plan_suite(suite, data_dir=tmp_path, host="", seconds_per_step=0.4)
    assert next(job for job in plan.jobs if job.case_id == "ridge-dusk").action == "save baseline"
    assert any("Legacy baselines were left untouched" in warning for warning in plan.warnings)


def test_normalize_host_adds_scheme_and_rejects_empty():
    assert normalize_base("127.0.0.1:9000") == "http://127.0.0.1:9000"
    assert normalize_base("http://127.0.0.1:9000/") == "http://127.0.0.1:9000"
    with pytest.raises(Exception):
        normalize_base("   ")


def test_default_config_host_is_empty(tmp_path):
    assert load_config(tmp_path)["comfy_host"] == ""


def test_python_sources_do_not_hardcode_the_example_host():
    for folder in (ROOT / "seedregress", ROOT / "scripts"):
        for path in folder.rglob("*.py"):
            assert "192.168.4.47" not in path.read_text(encoding="utf-8"), path


def test_websocket_handshake_reads_a_completion_frame():
    import base64
    import hashlib
    import socket
    import threading

    from seedregress.comfy import _WS_GUID

    ready = threading.Event()
    holder = {}

    def serve():
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        holder["port"] = server.getsockname()[1]
        ready.set()
        conn, _addr = server.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(4096)
        assert b"Upgrade: websocket" in data
        key = data.split(b"Sec-WebSocket-Key: ", 1)[1].split(b"\r\n", 1)[0].decode()
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        conn.sendall(response.encode())
        payload = b'{"type":"executing","data":{"node":null,"prompt_id":"p1"}}'
        header = bytes([0x81, len(payload)]) + payload
        conn.sendall(header)
        # Read the client's masked frame so the socket stays up until they finish.
        opcode, message = read_frame(conn)
        holder["reply"] = (opcode, message)
        conn.close()
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    client = ComfyClient(
        f"127.0.0.1:{holder['port']}",
        confirmed=True,
        install_id="install",
        use_websocket=True,
    )
    sock = client._open_websocket()
    assert sock is not None
    send_frame(sock, 0x9, b"ping")
    opcode, payload = read_frame(sock)
    sock.close()
    thread.join(timeout=2)
    assert opcode == 0x1
    assert b"p1" in payload
    assert holder["reply"][0] == 0x9
