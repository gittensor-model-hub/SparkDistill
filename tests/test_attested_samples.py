import json

from eval.attested_samples import (
    ATTESTED_SAMPLES_FILENAME,
    build_attested_samples_document,
    build_gsm8k_regression_entry,
    build_lm_eval_entry,
    build_triton_entry,
    read_attested_samples,
    verify_attested_eval_samples,
)
from eval.regression_sample import build_regression_sample, load_regression_problems


def _bound_attestation(bundle_dir, digest: str) -> dict:
    from eval.attestation import tdx_report_data

    return {
        "passed": True,
        "claims": {"eat_nonce": digest},
        "tdx": {"report_data": tdx_report_data(digest).hex()},
    }


def _claim_binding(bundle_dir, attestation):
    from eval.verify import check_claim_binding

    return check_claim_binding(bundle_dir, attestation)


def _tdx_binding(bundle_dir, attestation):
    from eval.verify import check_tdx_binding

    return check_tdx_binding(bundle_dir, attestation)


def test_verify_attested_samples_requires_gpu_and_tdx_bindings(tmp_path):
    responses = [
        {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
        for row in load_regression_problems()
    ]
    document = build_attested_samples_document(
        {"gsm8k": build_gsm8k_regression_entry(responses)}
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / ATTESTED_SAMPLES_FILENAME).write_text(json.dumps(document, indent=2))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 1.0}}))

    verified, issues = verify_attested_eval_samples(
        bundle,
        {"gsm8k": 1.0},
        {"gsm8k": 0.5},
        None,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert verified == set()
    assert any("GPU CC attestation" in issue for issue in issues)

    from proof.bundle import claim_sha256

    digest = claim_sha256(bundle)
    gpu_only = {"passed": True, "claims": {"eat_nonce": digest}}
    verified, issues = verify_attested_eval_samples(
        bundle,
        {"gsm8k": 1.0},
        {"gsm8k": 0.5},
        gpu_only,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert verified == set()
    assert any("TDX quote" in issue for issue in issues)


def test_verify_attested_lm_eval_and_triton_entries(tmp_path):
    lm_payload = {
        "results": {
            "humaneval": {"pass@1,none": 0.75},
        }
    }
    triton_report = {
        "summary": {"avg_composite": 0.5, "exec_pass_rate": 0.0, "avg_correctness": 0.0, "syntax_pass_rate": 1.0},
        "details": [
            {"level": 1, "composite_score": 0.5},
            {"level": "bugfix", "composite_score": 0.5},
        ],
    }
    document = build_attested_samples_document(
        {
            "humaneval": build_lm_eval_entry("humaneval", lm_payload, 0.75),
            "triton": build_triton_entry(triton_report),
        }
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / ATTESTED_SAMPLES_FILENAME).write_text(json.dumps(document, indent=2))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(
        json.dumps({"scores": {"humaneval": 0.75, "triton": 0.5, "triton_quick": 0.5}})
    )

    from proof.bundle import claim_sha256

    digest = claim_sha256(bundle)
    attestation = _bound_attestation(bundle, digest)

    verified, issues = verify_attested_eval_samples(
        bundle,
        {"humaneval": 0.75, "triton": 0.5, "triton_quick": 0.5},
        None,
        attestation,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert issues == []
    assert verified == {"humaneval", "triton"}


def test_read_attested_samples_wraps_legacy_gsm8k_file(tmp_path):
    sample = build_regression_sample(
        [
            {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
            for row in load_regression_problems()
        ]
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "gsm8k_regression_sample.json").write_text(json.dumps(sample, indent=2))

    wrapped = read_attested_samples(bundle)
    assert wrapped is not None
    assert "gsm8k" in wrapped["benchmarks"]
