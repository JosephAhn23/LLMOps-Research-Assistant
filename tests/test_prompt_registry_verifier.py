"""Prompt registry: skeptical verifier prompt loads."""

from pathlib import Path

from prompt_registry.registry import PromptRegistry


def test_load_skeptical_verifier_v1() -> None:
    root = Path(__file__).resolve().parents[1]
    reg = PromptRegistry(root / "prompt_registry" / "prompts")
    data = reg.load("skeptical_verifier", "v1")
    assert "skeptical" in data["system_prompt"].lower()
    assert "{{ draft_answer }}" in data["user_template"]
