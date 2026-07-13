import json

import teacher.generate as generate
from teacher.providers import Trajectory


class _SpyTeacher:
    """Minimal offline teacher that records nothing but a canned trajectory."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = "pinned"

    def generate(self, prompt: str, **_kwargs) -> Trajectory:
        return Trajectory(prompt=prompt, response="ok", provider=self.name, model=self.model)


def _install_spy(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_get_teacher(provider: str, model: str | None = None) -> _SpyTeacher:
        calls.append((provider, model))
        return _SpyTeacher(provider)

    monkeypatch.setattr(generate, "get_teacher", fake_get_teacher)
    return calls


def test_model_flag_is_not_forwarded_to_get_teacher(monkeypatch, tmp_path):
    # The --model flag is documented as ignored; forwarding it would make
    # get_teacher raise for any value that doesn't match a provider's pin.
    calls = _install_spy(monkeypatch)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')

    list(
        generate.generate_trajectories(
            prompts,
            ["anthropic", "openai"],
            max_tokens=16,
            temperature=0.0,
            limit=None,
            concurrency=1,
            thinking_budget=None,
        )
    )

    assert calls == [("anthropic", None), ("openai", None)]


def test_main_with_model_flag_does_not_crash(monkeypatch, tmp_path):
    # Regression: `--model gpt-5.6-sol` with the default two-provider basket used
    # to abort because get_teacher rejected the value on the anthropic branch.
    calls = _install_spy(monkeypatch)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"prompt": "hi"}\n')
    out = tmp_path / "out.jsonl"

    rc = generate.main(
        ["--prompts", str(prompts), "--out", str(out), "--model", "gpt-5.6-sol"]
    )

    assert rc == 0
    assert all(model is None for _, model in calls)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["provider"] for r in records} == {"anthropic", "openai"}
