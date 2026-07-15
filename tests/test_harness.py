import json

import eval.harness as harness


def _write_triton_sidecar(work_dir, scores, gpu_architecture=None):
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "triton.json").write_text(json.dumps({"scores": scores, "gpu_architecture": gpu_architecture}))


def test_main_preserves_triton_quick_in_claim(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    _write_triton_sidecar(
        work_dir,
        {
            "triton": 0.55,
            "triton_quick": 0.82,
            "triton_exec_pass_rate": 0.60,
            "triton_correctness": 0.50,
            "triton_syntax_pass_rate": 0.90,
        },
        gpu_architecture="hopper",
    )
    monkeypatch.setattr(harness, "run_harness", lambda *a, **k: {"triton": 0.55})

    out = tmp_path / "candidate.json"
    rc = harness.main(
        ["--checkpoint", "ckpt", "--benchmark", "triton", "--work-dir", str(work_dir), "--out", str(out)]
    )
    assert rc == 0

    payload = json.loads(out.read_text())
    scores = payload["scores"]
    assert scores["triton"] == 0.55
    assert scores["triton_quick"] == 0.82
    assert payload["gpu_architecture"] == "hopper"


def test_main_does_not_read_sidecar_when_triton_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        harness, "run_harness", lambda *a, **k: {"gsm8k": 0.6, "humaneval": 0.5}
    )
    out = tmp_path / "candidate.json"
    rc = harness.main(
        [
            "--checkpoint",
            "ckpt",
            "--benchmark",
            "gsm8k",
            "--benchmark",
            "humaneval",
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    scores = json.loads(out.read_text())["scores"]
    assert scores == {"gsm8k": 0.6, "humaneval": 0.5}
