import json

from teacher.filter import (
    drop_reason,
    filter_trajectories,
    looks_like_refusal,
    read_trajectories,
)


def _traj(prompt="What is 2 + 2?", response="4", **extra):
    return {"prompt": prompt, "response": response, **extra}


def test_keeps_a_well_formed_trajectory():
    assert drop_reason(_traj()) is None


def test_drops_non_object_records_as_malformed():
    assert drop_reason("not a dict") == "malformed"
    assert drop_reason(["also", "not"]) == "malformed"
    assert drop_reason(None) == "malformed"


def test_drops_empty_or_missing_prompt():
    assert drop_reason(_traj(prompt="")) == "empty_prompt"
    assert drop_reason(_traj(prompt="   \n\t")) == "empty_prompt"
    assert drop_reason({"response": "4"}) == "empty_prompt"


def test_drops_empty_or_whitespace_response():
    assert drop_reason(_traj(response="")) == "empty_response"
    assert drop_reason(_traj(response="   ")) == "empty_response"
    assert drop_reason({"prompt": "hi"}) == "empty_response"


def test_length_bounds_are_opt_in():
    short = _traj(response="hi")
    # Disabled by default.
    assert drop_reason(short) is None
    assert drop_reason(short, min_response_chars=8) == "too_short"
    assert drop_reason(_traj(response="x" * 100), max_response_chars=10) == "too_long"


def test_refusal_detection_is_anchored_to_the_response_head():
    assert looks_like_refusal("I cannot help with that request.")
    assert looks_like_refusal("  Sorry, I can't do that.  ")
    # A legitimate answer that merely mentions refusals must survive.
    assert not looks_like_refusal("Here is why a model might say it cannot help: ...")
    assert not looks_like_refusal("The kernel cannot assist the scheduler here; instead ...")


def test_refusals_dropped_by_default_but_keepable():
    refusal = _traj(response="I'm sorry, but I can't provide that.")
    assert drop_reason(refusal) == "refusal"
    assert drop_reason(refusal, drop_refusals=False) is None


def test_filter_trajectories_reports_stats_and_preserves_order():
    records = [
        _traj(prompt="a", response="alpha answer"),
        "corrupt-line",
        _traj(prompt="b", response=""),
        _traj(prompt="c", response="I cannot assist with that."),
        _traj(prompt="d", response="delta answer"),
    ]
    kept, stats = filter_trajectories(records)

    assert [r["prompt"] for r in kept] == ["a", "d"]
    assert stats.total == 5
    assert stats.kept == 2
    assert stats.dropped == 3
    assert stats.dropped_by_reason == {
        "malformed": 1,
        "empty_response": 1,
        "refusal": 1,
    }


def test_dedupe_prompts_keeps_first_occurrence_only():
    records = [
        _traj(prompt="Same  Question", response="first"),
        _traj(prompt="same question", response="second (dup, case/space-insensitive)"),
        _traj(prompt="different", response="third"),
    ]
    kept, stats = filter_trajectories(records, dedupe_prompts=True)

    assert [r["response"] for r in kept] == ["first", "third"]
    assert stats.dropped_by_reason["duplicate_prompt"] == 1


def test_dedupe_is_off_by_default():
    records = [_traj(prompt="q", response="one"), _traj(prompt="q", response="two")]
    kept, _ = filter_trajectories(records)
    assert len(kept) == 2


def test_read_trajectories_surfaces_bad_lines_as_malformed(tmp_path):
    path = tmp_path / "traj.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_traj(prompt="good", response="kept answer")),
                "",  # blank line skipped
                "{not valid json",  # surfaced as raw text -> malformed
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kept, stats = filter_trajectories(read_trajectories(path))

    assert [r["prompt"] for r in kept] == ["good"]
    assert stats.total == 2
    assert stats.dropped_by_reason["malformed"] == 1


def test_stats_to_dict_is_json_serializable():
    _, stats = filter_trajectories([_traj(), "bad"])
    payload = json.loads(json.dumps(stats.to_dict()))
    assert payload["total"] == 2
    assert payload["kept"] == 1
    assert payload["dropped_by_reason"]["malformed"] == 1
