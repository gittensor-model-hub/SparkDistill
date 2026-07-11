import json
from pathlib import Path
from types import SimpleNamespace

import eval.registry_gate as registry_gate
from eval.registry_gate import (
    check_registry_duplicates,
    close_dataset_pr,
    gate_registry_pr,
    is_dataset_track_pr,
    merge_eligible,
    parse_added_registry_lines,
    reward_eligible,
    update_pr_dataset_label,
    validate_append_only_registry,
    validate_changed_paths,
    validate_registry_entry,
)


def _entry(**overrides):
    base = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/sparkproof-triton-v0",
        "trajectories_sha256": "a" * 64,
        "rows_total": 128,
        "dataset_version": "triton-distill-v0.2",
    }
    base.update(overrides)
    return base


def test_parse_added_registry_lines():
    base = ""
    head = json.dumps(_entry()) + "\n"
    added = parse_added_registry_lines(base, head)
    assert len(added) == 1
    assert added[0]["miner"] == "alice"


def test_dataset_track_checkbox_must_be_checked():
    assert is_dataset_track_pr("- [x] **Dataset track submission**")
    assert is_dataset_track_pr("- [X] **Dataset track submission**")
    assert not is_dataset_track_pr("- [ ] **Dataset track submission**")
    assert not is_dataset_track_pr("- [x] **Training/evaluation improvement**")


def test_pr_template_checkbox_matches_gate_parser():
    template = Path(".github/pull_request_template.md").read_text(encoding="utf-8")
    checked = template.replace(
        "- [ ] **Dataset track submission**",
        "- [x] **Dataset track submission**",
        1,
    )
    assert is_dataset_track_pr(checked)


def test_dataset_track_rejects_changes_outside_registry():
    assert validate_changed_paths(["datasets/registry.jsonl"]) == []
    issues = validate_changed_paths(["datasets/registry.jsonl", "eval/registry_gate.py"])
    assert any("may only change" in issue for issue in issues)


def test_registry_must_only_append_to_latest_base():
    existing = json.dumps(_entry()) + "\n"
    appended = existing + json.dumps(_entry(miner="bob", trajectories_sha256="b" * 64)) + "\n"
    assert validate_append_only_registry(existing, appended) == []
    issues = validate_append_only_registry(existing, appended.splitlines()[1] + "\n")
    assert any("append-only" in issue for issue in issues)


def test_reward_and_merge_eligibility_require_dataset_xs_or_above():
    for label in ("dataset:xs", "dataset:s", "dataset:m", "dataset:l", "dataset:xl"):
        report = {"verified": True, "label": label}
        assert reward_eligible(report)
        assert merge_eligible(report)
    assert not reward_eligible({"verified": True, "label": "dataset:none"})
    assert not merge_eligible({"verified": False, "label": "dataset:l"})


def test_validate_registry_entry_requires_fields():
    issues = validate_registry_entry({"miner": "alice"})
    assert any("hf_url" in issue for issue in issues)


def test_check_registry_duplicates_rejects_repeat_sha():
    existing = [_entry()]
    issues = check_registry_duplicates(existing, [_entry(miner="bob")])
    assert any("duplicate trajectories_sha256" in issue for issue in issues)


def test_gate_registry_pr_rejects_multi_line_append():
    entry = json.dumps(_entry())
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=entry + "\n" + entry + "\n",
        sparkproof_root=__import__("pathlib").Path("."),
    )
    assert report["verified"] is False
    assert any("exactly one" in issue for issue in report["issues"])


def test_gate_registry_pr_rejects_schema_before_hf(monkeypatch):
    entry = json.dumps(_entry(rows_total=0))
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=entry + "\n",
        sparkproof_root=__import__("pathlib").Path("."),
    )
    assert report["verified"] is False
    assert report["submissions"][0]["issues"]


def test_gate_registry_pr_requires_checked_dataset_track():
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_entry()) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [ ] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
    )
    assert report["verified"] is False
    assert report["label"] == "dataset:REJECT"
    assert any("Dataset track submission" in issue for issue in report["issues"])


def test_verified_sub_threshold_dataset_is_not_merge_eligible(monkeypatch):
    monkeypatch.setattr(
        registry_gate,
        "gate_registry_submission",
        lambda *args, **kwargs: {
            "verified": True,
            "label": "dataset:none",
            "rows_total": 24,
            "issues": [],
        },
    )
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_entry(rows_total=24)) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [x] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
    )
    assert report["verified"] is True
    assert report["reward_eligible"] is False
    assert report["merge_eligible"] is False
    assert report["label"] == "dataset:none"
    assert any("25 verified rows" in issue for issue in report["issues"])


def test_verified_dataset_xs_is_merge_eligible(monkeypatch):
    monkeypatch.setattr(
        registry_gate,
        "gate_registry_submission",
        lambda *args, **kwargs: {
            "verified": True,
            "label": "dataset:xs",
            "rows_total": 25,
            "issues": [],
        },
    )
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_entry(rows_total=25)) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [x] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
    )
    assert report["reward_eligible"] is True
    assert report["merge_eligible"] is True
    assert report["issues"] == []


def test_verified_dataset_s_is_merge_eligible(monkeypatch):
    monkeypatch.setattr(
        registry_gate,
        "gate_registry_submission",
        lambda *args, **kwargs: {
            "verified": True,
            "label": "dataset:s",
            "rows_total": 50,
            "issues": [],
        },
    )
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_entry(rows_total=50)) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [x] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
    )
    assert report["reward_eligible"] is True
    assert report["merge_eligible"] is True
    assert report["issues"] == []


def test_update_pr_dataset_label_replaces_stale_label(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["gh", "api"] and "--method" not in command:
            return SimpleNamespace(returncode=0, stdout="dataset:none\nneeds-review\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(registry_gate.subprocess, "run", fake_run)
    assert update_pr_dataset_label(42, "dataset:s") == []
    assert [
        "gh",
        "api",
        "--method",
        "DELETE",
        "repos/{owner}/{repo}/issues/42/labels/dataset:none",
    ] in calls
    assert [
        "gh",
        "api",
        "--method",
        "POST",
        "repos/{owner}/{repo}/issues/42/labels",
        "-f",
        "labels[]=dataset:s",
    ] in calls


def test_close_dataset_pr_posts_gate_comment(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(registry_gate.subprocess, "run", fake_run)
    assert close_dataset_pr(7, ["forged sha256"]) == []
    assert calls[0][:4] == ["gh", "pr", "close", "7"]
    assert "forged sha256" in calls[0][-1]
