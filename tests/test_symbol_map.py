from pathlib import Path

from context_engineering.symbol_map import generate_symbol_map_text


def test_symbol_map_includes_class_and_method(tmp_path: Path) -> None:
    pkg = tmp_path / "my_pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "class Foo:\n    def bar(self):\n        pass\n\ndef spam():\n    pass\n",
        encoding="utf-8",
    )
    text = generate_symbol_map_text(
        tmp_path,
        package_roots=["my_pkg"],
        max_files=20,
        max_lines=100,
    )
    assert "class Foo" in text
    assert "def spam" in text or "spam" in text
