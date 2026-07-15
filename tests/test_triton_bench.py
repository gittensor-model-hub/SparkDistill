import json
import os
import time

import pytest

from eval.benchmarks import BENCHMARKS, run_benchmark
from eval.score import score
from eval.triton_bench import detect_gpu_architecture, latest_report, run_tritonbench, serve_checkpoint, summary_scores


def _report(composite=0.71, exec_pass=0.65, correctness=0.7, syntax=0.9, details=None):
    return {
        "summary": {
            "avg_composite": composite,
            "exec_pass_rate": exec_pass,
            "avg_correctness": correctness,
            "syntax_pass_rate": syntax,
            "avg_api_modernity": 0.8,
            "avg_perf_awareness": 0.5,
            "avg_gen_time_s": 12.0,
        },
        "num_problems": 20,
        **({"details": details} if details is not None else {}),
    }


def test_summary_scores_flattens_headline_and_submetrics():
    scores = summary_scores(_report())
    assert scores["triton"] == 0.71
    assert scores["triton_exec_pass_rate"] == 0.65
    assert scores["triton_correctness"] == 0.7
    assert scores["triton_syntax_pass_rate"] == 0.9


def test_summary_scores_empty_report_is_zero():
    assert summary_scores({})["triton"] == 0.0


def test_summary_scores_quick_subset_from_details():
    details = [
        {"level": 1, "composite_score": 0.9},
        {"level": "bugfix", "composite_score": 0.7},
        {"level": 4, "composite_score": 0.1},
    ]
    scores = summary_scores(_report(composite=0.5667, details=details))
    # triton_quick covers only the level-1 + bugfix subset a quick re-run sees.
    assert scores["triton_quick"] == pytest.approx(0.8)
    assert scores["triton"] == pytest.approx(0.5667)


def test_summary_scores_quick_falls_back_to_headline_without_details():
    scores = summary_scores(_report(composite=0.71))
    assert scores["triton_quick"] == 0.71


def _write_report(path, report, mtime_ns):
    path.write_text(json.dumps(report))
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_latest_report_picks_newest(tmp_path):
    now = time.time_ns()
    _write_report(tmp_path / "tritonbench_m_20260101_000000.json", _report(composite=0.1), now - 1_000_000)
    _write_report(tmp_path / "tritonbench_m_20260201_000000.json", _report(composite=0.9), now)
    assert latest_report(tmp_path)["summary"]["avg_composite"] == 0.9


def test_latest_report_ignores_stale_report_for_other_model(tmp_path):
    # "zeta" sorts after "alpha" lexicographically but is the older run — mtime,
    # not the model-name-first filename, must decide which report is newest.
    now = time.time_ns()
    _write_report(tmp_path / "tritonbench_zeta_20260101_000000.json", _report(composite=0.1), now - 1_000_000)
    _write_report(tmp_path / "tritonbench_alpha_20260201_000000.json", _report(composite=0.9), now)
    assert latest_report(tmp_path)["summary"]["avg_composite"] == 0.9


def test_latest_report_newer_than_excludes_preexisting(tmp_path):
    stamp = time.time_ns()
    _write_report(tmp_path / "tritonbench_m_1.json", _report(composite=0.1), stamp - 1_000_000)
    with pytest.raises(FileNotFoundError):
        latest_report(tmp_path, newer_than_ns=stamp)
    _write_report(tmp_path / "tritonbench_m_2.json", _report(composite=0.9), stamp + 1_000_000)
    assert latest_report(tmp_path, newer_than_ns=stamp)["summary"]["avg_composite"] == 0.9


def test_run_tritonbench_config_matches_run_depth(tmp_path, monkeypatch):
    import eval.triton_bench as tb

    commands = []

    def fake_run(command, cwd=None, check=None, timeout=None):
        commands.append(command)
        path = tmp_path / "results" / f"tritonbench_m_{len(commands)}.json"
        # Stamp explicitly past the run's start — write_text alone can land on a
        # coarser filesystem tick than the nanosecond clock the filter uses.
        _write_report(path, _report(), time.time_ns() + 1_000_000)

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    run_tritonbench("http://x/v1", "m", tmp_path / "results", tb._QUICK_LEVELS, bench_root=tmp_path)
    run_tritonbench("http://x/v1", "m", tmp_path / "results", tb._FULL_LEVELS, bench_root=tmp_path)

    assert tb._QUICK_CONFIG in commands[0]
    assert tb._FULL_CONFIG in commands[1]


def test_serve_checkpoint_passes_served_model_name(monkeypatch):
    import eval.triton_bench as tb

    captured = {}

    class FakeProc:
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, stdout=None, stderr=None):
        captured["command"] = command
        return FakeProc()

    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tb, "_endpoint_ready", lambda endpoint: True)

    with serve_checkpoint("/models/ckpt", served_model_name="ckpt") as endpoint:
        assert endpoint.endswith("/v1")
    assert "--served-model-name" in captured["command"]
    assert "ckpt" in captured["command"]
    assert "--seed" in captured["command"]
    assert "--no-enable-prefix-caching" in captured["command"]


def test_triton_registered_in_basket():
    assert "triton" in BENCHMARKS
    assert BENCHMARKS["triton"].metric == "avg_composite"


def test_run_benchmark_dispatches_triton_to_adapter(tmp_path, monkeypatch):
    calls = {}

    def fake_run(model_path, output_dir, limit=None, endpoint=None):
        calls["args"] = (model_path, output_dir, limit)
        return 0.42

    import eval.triton_bench as tb

    monkeypatch.setattr(tb, "run_triton_benchmark", fake_run)
    result = run_benchmark(BENCHMARKS["triton"], "outputs/student", tmp_path, limit=5)
    assert result == 0.42
    assert calls["args"] == ("outputs/student", tmp_path, 5)


def test_score_tiers_triton_improvement():
    candidate = {"triton": 0.71, "gsm8k": 0.88}
    frontier = {"triton": 0.60, "gsm8k": 0.88}
    report = score(candidate, frontier)
    assert report["label"] == "eval:XL"  # 18.3% relative improvement on triton
    assert report["best_benchmark"] == "triton"


def test_score_flags_triton_regression():
    candidate = {"triton": 0.50, "gsm8k": 0.90}
    frontier = {"triton": 0.60, "gsm8k": 0.88}
    report = score(candidate, frontier)
    assert "regression-triton" in report["regressions"]
    assert report["label"] == "eval:REJECT"


def test_locate_results_file_prefers_exact_then_date_suffixed(tmp_path):
    from eval.benchmarks import _locate_results_file

    exact = tmp_path / "gsm8k.json"
    dated_old = tmp_path / "gsm8k_2026-07-11T00-00-00.json"
    dated_new = tmp_path / "gsm8k_2026-07-11T01-00-00.json"
    dated_old.write_text("{}")
    os.utime(dated_old, ns=(1, 1))
    dated_new.write_text("{}")
    assert _locate_results_file(exact) == dated_new
    exact.write_text("{}")
    assert _locate_results_file(exact) == exact


def test_detect_gpu_architecture_env_override_wins(monkeypatch):
    import eval.triton_bench as tb

    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "hopper-h200")

    def fail_run(*a, **k):
        raise AssertionError("nvidia-smi should not run when the env override is set")

    monkeypatch.setattr(tb.subprocess, "run", fail_run)
    assert detect_gpu_architecture() == "hopper"


def test_detect_gpu_architecture_parses_nvidia_smi_name(monkeypatch):
    import eval.triton_bench as tb

    monkeypatch.delenv("SPARKDISTILL_GPU_ARCHITECTURE", raising=False)

    class Result:
        returncode = 0
        stdout = "NVIDIA H100 80GB HBM3\n"

    monkeypatch.setattr(tb.subprocess, "run", lambda *a, **k: Result())
    assert detect_gpu_architecture() == "hopper"


def test_detect_gpu_architecture_returns_none_without_nvidia_smi(monkeypatch):
    import eval.triton_bench as tb

    monkeypatch.delenv("SPARKDISTILL_GPU_ARCHITECTURE", raising=False)

    def fake_run(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    assert detect_gpu_architecture() is None


def test_run_triton_benchmark_records_gpu_architecture(tmp_path, monkeypatch):
    import eval.triton_bench as tb

    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "blackwell")
    monkeypatch.setattr(tb, "run_tritonbench", lambda *a, **k: _report())

    headline = tb.run_triton_benchmark("outputs/student", tmp_path, endpoint="http://x/v1")
    assert headline == 0.71
    detail = json.loads((tmp_path / "triton.json").read_text())
    assert detail["gpu_architecture"] == "blackwell"


def test_extract_metric_handles_lm_eval_filter_suffixes():
    from eval.benchmarks import _extract_metric

    assert _extract_metric({"exact_match": 0.9}, "exact_match") == 0.9
    assert (
        _extract_metric({"exact_match,flexible-extract": 0.8, "exact_match,strict-match": 0.7}, "exact_match") == 0.7
    )
    assert _extract_metric({"acc,none": 0.6}, "acc") == 0.6
