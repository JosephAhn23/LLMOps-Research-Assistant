"""Utilities for loading and listing versioned prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


class PromptRegistry:
    """
    Filesystem-backed prompt registry.

    Prompt files follow: `{name}_{version}.yaml` inside the prompts directory.
    """

    def __init__(self, prompts_dir: str | Path = "prompt_registry/prompts") -> None:
        self.prompts_dir = Path(prompts_dir)

    def load(self, name: str, version: str) -> Dict[str, Any]:
        """
        Load one prompt version.

        Args:
            name: Prompt family name, e.g. "rag_synthesizer".
            version: Version string, e.g. "v1".

        Returns:
            Parsed YAML as a dictionary.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
            ValueError: If YAML is malformed or metadata does not match request.
        """
        prompt_path = self.prompts_dir / f"{name}_{version}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        with prompt_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid prompt file format: {prompt_path}")

        if data.get("name") != name or data.get("version") != version:
            raise ValueError(
                f"Prompt metadata mismatch in {prompt_path}: "
                f"expected name={name}, version={version}"
            )

        return data

    def list_versions(self, name: str) -> List[str]:
        """
        List available versions for a prompt family.

        Args:
            name: Prompt family name.

        Returns:
            Sorted version strings, for example ["v1", "v2"].
        """
        if not self.prompts_dir.exists():
            return []

        versions: List[str] = []
        for file_path in sorted(self.prompts_dir.glob(f"{name}_v*.yaml")):
            stem = file_path.stem
            # Example stem: rag_synthesizer_v1
            maybe_version = stem.replace(f"{name}_", "", 1)
            if maybe_version.startswith("v"):
                versions.append(maybe_version)

        return sorted(versions, key=self._version_sort_key)

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, str]:
        """Sort versions numerically when possible, lexicographically otherwise."""
        numeric = version[1:] if version.startswith("v") else version
        if numeric.isdigit():
            return (int(numeric), version)
        return (10**9, version)

