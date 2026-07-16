import eval.harness as harness
import eval.model_server as model_server
from eval.benchmarks import BENCHMARKS, run_benchmark


def test_eval_session_uses_student_endpoint_env(monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_STUDENT_ENDPOINT", "http://127.0.0.1:8000/v1")

    with model_server.eval_session("outputs/qwen3.5-4b-phase1") as session:
        assert session is not None
        assert session.endpoint == "http://127.0.0.1:8000/v1"
        assert session.model_name == "qwen3.5-4b-phase1"


def test_eval_session_serve_starts_vllm_once(monkeypatch):
    monkeypatch.delenv("SPARKDISTILL_STUDENT_ENDPOINT", raising=False)
    calls = []

    class FakeCtx:
        def __enter__(self):
            return "http://127.0.0.1:8000/v1"

        def __exit__(self, *exc):
            return False

    def fake_serve(model_path, served_model_name=None):
        calls.append((model_path, served_model_name))
        return FakeCtx()

    import eval.triton_bench as tb

    monkeypatch.setattr(tb, "serve_checkpoint", fake_serve)

    with model_server.eval_session("outputs/student", serve=True) as session:
        assert session is not None
        assert session.endpoint.endswith("/v1")
        assert session.model_name == "student"
    assert calls == [("outputs/student", "student")]


def test_eval_session_yields_none_without_serve(monkeypatch):
    monkeypatch.delenv("SPARKDISTILL_STUDENT_ENDPOINT", raising=False)
    monkeypatch.delenv("SPARKDISTILL_EVAL_SERVE", raising=False)

    with model_server.eval_session("outputs/student") as session:
        assert session is None


def test_run_benchmark_uses_local_completions_when_endpoint_set(tmp_path, monkeypatch):
    import eval.benchmarks as benchmarks

    captured = []

    def fake_run(command, check=None):
        captured.append(command)
        result_path = tmp_path / "gsm8k.json"
        result_path.write_text(
            '{"results": {"gsm8k": {"exact_match": 0.42}}}'
        )

    monkeypatch.setattr(benchmarks.subprocess, "run", fake_run)

    score = run_benchmark(
        BENCHMARKS["gsm8k"],
        "outputs/student",
        tmp_path,
        endpoint="http://127.0.0.1:8000/v1",
        model_name="student",
    )
    assert score == 0.42
    assert captured[0][2] == "local-completions"
    model_args = captured[0][4]
    assert "base_url=http://127.0.0.1:8000/v1/completions" in model_args
    assert "tokenizer=outputs/student" in model_args
    assert "model=student" in model_args


def test_run_harness_reuses_shared_endpoint(tmp_path, monkeypatch):
    calls = []

    class FakeSession:
        endpoint = "http://127.0.0.1:8000/v1"
        model_name = "student"

    class FakeCtx:
        def __enter__(self):
            return FakeSession()

        def __exit__(self, *exc):
            return False

    def fake_eval_session(model_path, serve=False):
        return FakeCtx()

    def fake_run_benchmark(benchmark, model_path, output_dir, limit=None, *, endpoint=None, model_name=None):
        calls.append((benchmark.key, endpoint, model_name))
        return 0.5

    monkeypatch.setattr(harness, "eval_session", fake_eval_session)
    monkeypatch.setattr(harness, "run_benchmark", fake_run_benchmark)

    scores = harness.run_harness(
        "outputs/student",
        ["gsm8k", "humaneval"],
        tmp_path,
        serve=True,
    )
    assert scores == {"gsm8k": 0.5, "humaneval": 0.5}
    assert calls == [
        ("gsm8k", "http://127.0.0.1:8000/v1", "student"),
        ("humaneval", "http://127.0.0.1:8000/v1", "student"),
    ]
