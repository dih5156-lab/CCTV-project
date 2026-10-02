import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.ai._temporal_worker import TemporalWorker


@pytest.fixture
def command(tmp_path):
    # 실제 파이프/프로세스로 통신·강제 종료를 시험하되 GPU는 필요하지 않다.
    script = tmp_path / "child.py"
    script.write_text('''import sys,json,os,time
for line in sys.stdin:
    r=json.loads(line); action=r['video']
    if action=='timeout': time.sleep(5)
    if action=='exit': sys.exit(1)
    if action=='invalid':
        print('TCN_RESULT not-json',flush=True); continue
    out={'request_id':r['request_id'], 'stdout':str(os.getpid())+':'+action}
    if action=='mismatch': out['request_id']=-1
    if action=='error': out['error']='CUDA trial error'
    print('TCN_RESULT '+json.dumps(out),flush=True)
''')
    model = tmp_path / "model.pt"
    model.write_text("model")
    return [sys.executable, str(script), "--model", str(model),
            "--pose-model", str(model), "--video", "ok"]


def test_worker_reuses_process_and_reloads_replaced_model(command):
    worker = TemporalWorker()
    try:
        first = worker.run(command, timeout=3).stdout
        assert worker.run(command, timeout=3).stdout == first
        Path(command[3]).write_text("replacement model")
        assert worker.run(command, timeout=3).stdout != first
    finally:
        worker.close()
    assert worker._process is None


def test_verifier_close_reaps_cached_child_and_allows_next_start(command):
    from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier

    verifier = FallDataAuxVerifier(FallDataAuxConfig())
    try:
        first = verifier._temporal_worker.run(command, timeout=3).stdout
        child = verifier._temporal_worker._process
        verifier.close()
        assert child.poll() is not None
        assert verifier._temporal_worker._process is None
        verifier.close()
        second = verifier._temporal_worker.run(command, timeout=3).stdout
        assert first != second
    finally:
        verifier._temporal_worker.close()


@pytest.mark.parametrize("action", ["timeout", "exit", "invalid", "mismatch", "error"])
def test_failed_worker_is_discarded_and_next_call_recovers(command, action):
    worker = TemporalWorker()
    try:
        first = worker.run(command, timeout=3).stdout
        with pytest.raises((subprocess.TimeoutExpired, RuntimeError, ValueError)):
            worker.run([*command[:-1], action], timeout=0.2)
        assert worker._process is None
        assert worker.run(command, timeout=3).stdout != first
    finally:
        worker.close()


def test_worker_lock_wait_is_bounded(command):
    worker = TemporalWorker()
    worker._lock.acquire()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            worker.run(command, timeout=0.01)
    finally:
        worker._lock.release()
        worker.close()


def test_cli_worker_loads_once_and_keeps_requests_separate(monkeypatch, capsys):
    from scripts.inference import smoke_fall_temporal_model as cli

    args = SimpleNamespace(feature_json=None, output_feature_json=None)
    resources = object()
    loads, videos = [], []
    monkeypatch.setattr(cli, "_load_resources", lambda args: loads.append(1) or resources)
    def infer(args, loaded):
        assert loaded is resources
        videos.append(str(args.video))
        print("prediction: [0]")
        return 0
    monkeypatch.setattr(cli, "_infer", infer)
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        '{"request_id":1,"video":"a.mp4"}\n{"request_id":2,"video":"b.mp4"}\n'))
    monkeypatch.setattr(cli.select, "select", lambda *args: ([sys.stdin], [], []))
    assert cli._serve(args) == 0
    responses = [json.loads(line.removeprefix("TCN_RESULT "))
                 for line in capsys.readouterr().out.splitlines()]
    assert [r["request_id"] for r in responses] == [1, 2]
    assert loads == [1]
    assert videos == ["a.mp4", "b.mp4"]


def test_aux_worker_option_preserves_parser_and_metrics(monkeypatch, tmp_path):
    from src.core.ai import _falldata_aux as aux

    path = tmp_path / "dependency"
    path.touch()
    monkeypatch.setenv("FALLDATA_AUX_TEMPORAL_WORKER_ENABLED", "true")
    assert aux.FallDataAuxConfig.from_env().temporal_worker_enabled is True
    assert aux.FallDataAuxConfig().temporal_worker_enabled is False
    verifier = aux.FallDataAuxVerifier(aux.FallDataAuxConfig(
        temporal_worker_enabled=True, temporal_python=path,
        temporal_compare_model_path=path, temporal_pose_model_path=path,
    ))
    calls = []
    def run(command, *, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=(
            'prediction: [1]\nfall_probability: 0.9\nthreshold: 0.7\n'
            'frames_with_pose: 30\nwindow_inferences: 1\n'))
    monkeypatch.setattr(verifier._temporal_worker, "run", run)
    result = verifier._run_temporal_compare_model(tmp_path / "video.mp4")
    assert result["confirmed"] is True
    assert len(calls) == 1
    assert verifier.temporal_metrics_snapshot()["worker_completed"] == 1
    assert "process_calls" not in verifier.temporal_metrics_snapshot()
    from src.services.processor_metrics import ProcessorStatsCollector

    stats = verifier.temporal_metrics_snapshot()
    collector = ProcessorStatsCollector(SimpleNamespace(
        get_stats=lambda: {"falldata_temporal_stats": stats},
    ))
    samples = {sample.name: sample.value for family in collector.collect()
               for sample in family.samples}
    assert samples["cctv_falldata_temporal_worker_completed_total"] == 1


def test_cli_worker_stops_after_inference_error(monkeypatch, capsys):
    from scripts.inference import smoke_fall_temporal_model as cli

    monkeypatch.setattr(cli, "_load_resources", lambda args: object())
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError("simulated CUDA error")
    monkeypatch.setattr(cli, "_infer", fail)
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        '{"request_id":1,"video":"a.mp4"}\n{"request_id":2,"video":"b.mp4"}\n'))
    monkeypatch.setattr(cli.select, "select", lambda *args: ([sys.stdin], [], []))
    assert cli._serve(SimpleNamespace(feature_json=None, output_feature_json=None)) == 1
    response = json.loads(capsys.readouterr().out.removeprefix("TCN_RESULT "))
    assert "error" in response
    assert calls == [1]
