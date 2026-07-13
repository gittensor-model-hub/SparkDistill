import json

from teacher.report import read_trajectories, summarize


def _traj(prompt="q", response="a", provider="anthropic", model="claude-fable-5", **extra):
    return {"prompt": prompt, "response": response, "provider": provider, "model": model, **extra}


def test_counts_totals_and_valid():
    report = summarize([_traj(), _traj(), "corrupt"])
    assert report.total == 3
    assert report.malformed == 1
    assert report.valid == 2


def test_reasoning_capture_rate_overall():
    records = [
        _traj(reasoning="because ..."),
        _traj(reasoning="   "),  # whitespace-only reasoning does not count
        _traj(),  # no reasoning key
        _traj(reasoning="another trace"),
    ]
    report = summarize(records)
    assert report.with_reasoning == 2
    assert report.reasoning_capture_rate == 0.5


def test_reasoning_capture_rate_per_provider():
    records = [
        _traj(provider="anthropic", reasoning="trace"),
        _traj(provider="anthropic"),
        _traj(provider="openai"),
        _traj(provider="openai"),
    ]
    by_provider = summarize(records).to_dict()["reasoning_capture_by_provider"]
    assert by_provider["anthropic"] == {"with_reasoning": 1, "total": 2, "rate": 0.5}
    assert by_provider["openai"] == {"with_reasoning": 0, "total": 2, "rate": 0.0}


def test_provider_and_model_breakdown():
    records = [
        _traj(provider="anthropic", model="claude-fable-5"),
        _traj(provider="openai", model="gpt-5.6"),
        _traj(provider="openai", model="gpt-5.6"),
    ]
    report = summarize(records)
    assert dict(report.by_provider) == {"anthropic": 1, "openai": 2}
    assert dict(report.by_model) == {"claude-fable-5": 1, "gpt-5.6": 2}


def test_missing_provider_bucketed_as_unknown():
    report = summarize([{"prompt": "q", "response": "a"}])
    assert dict(report.by_provider) == {"unknown": 1}


def test_empty_prompt_and_response_counters():
    records = [
        _traj(prompt="  ", response="ok"),
        _traj(prompt="q", response="   "),
        _traj(prompt="q", response="ok"),
    ]
    report = summarize(records)
    assert report.empty_prompts == 1
    assert report.empty_responses == 1


def test_length_stats_ignore_empty_responses():
    records = [_traj(response="abcd"), _traj(response="ab"), _traj(response="")]
    stats = summarize(records).to_dict()["response_chars"]
    assert stats == {"count": 2, "min": 2, "mean": 3.0, "median": 3, "max": 4}


def test_length_stats_are_none_when_no_samples():
    report = summarize([_traj(response="")])
    assert report.to_dict()["response_chars"] is None
    assert report.to_dict()["reasoning_chars"] is None


def test_with_system_counter():
    records = [_traj(system="You are helpful."), _traj(system=""), _traj()]
    assert summarize(records).with_system == 1


def test_report_is_json_serializable():
    report = summarize([_traj(reasoning="x"), "bad"])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["total"] == 2
    assert payload["malformed"] == 1
    assert payload["reasoning_capture_rate"] == 1.0  # 1 with-reasoning / 1 valid


def test_read_trajectories_skips_blanks_and_surfaces_bad_lines(tmp_path):
    path = tmp_path / "traj.jsonl"
    path.write_text(
        "\n".join([json.dumps(_traj()), "", "{bad json"]) + "\n",
        encoding="utf-8",
    )
    report = summarize(read_trajectories(path))
    assert report.total == 2
    assert report.malformed == 1
    assert report.valid == 1
