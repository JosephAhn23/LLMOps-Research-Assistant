"""SageMaker processing step helper script."""
import json
from pathlib import Path


def main() -> None:
    input_dir = Path("/opt/ml/processing/input")
    output_dir = Path("/opt/ml/processing/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass-through example to keep pipeline runnable; customize for real cleaning.
    in_file = input_dir / "train.jsonl"
    out_file = output_dir / "train.jsonl"
    if in_file.exists():
        rows = [json.loads(line) for line in in_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_file.write_text("
".join(json.dumps(r) for r in rows), encoding="utf-8")
    else:
        out_file.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
