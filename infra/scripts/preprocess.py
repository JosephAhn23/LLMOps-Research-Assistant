from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    input_path = Path("/opt/ml/processing/input/train.jsonl")
    output_dir = Path("/opt/ml/processing/train")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Expected input dataset at {input_path}")

    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    cleaned = []
    for row in rows:
        if "query" in row and "positive" in row:
            cleaned.append({"query": row["query"], "positive": row["positive"]})

    out_file = output_dir / "train.jsonl"
    out_file.write_text("\n".join(json.dumps(r) for r in cleaned), encoding="utf-8")
    print(f"Wrote {len(cleaned)} training rows to {out_file}")


if __name__ == "__main__":
    main()
