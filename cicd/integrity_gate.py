"""
CI gate: scientific integrity scan on a sample model output.

Complements ``cicd/ragas_gate.py`` (RAGAS thresholds). This script fails the build
when hedging / assumption / uncited-claim rules trip (configurable).

Usage:
    python cicd/integrity_gate.py --text-file outputs/sample_answer.txt
    python cicd/integrity_gate.py --text "The result is 42 without proof."
    python cicd/integrity_gate.py --text-file answer.txt --chunks-file chunks.json

``chunks.json`` format: list of {\"text\": \"...\", \"source\": \"...\"} in order
so ``[source_1]`` maps to index 0 (ingestion-aligned).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_chunks(path: Optional[str]) -> Optional[List[dict[str, Any]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.error("Chunks file not found: %s", path)
        sys.exit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        logger.error("chunks-file must be a JSON array of chunk objects")
        sys.exit(1)
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scientific integrity CI gate")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text-file", help="Path to UTF-8 text file to scan")
    g.add_argument("--text", help="Literal answer text to scan")
    p.add_argument(
        "--chunks-file",
        default=None,
        help="Optional JSON list of chunks for citation substantiation checks",
    )
    p.add_argument(
        "--fail-on-hedging",
        action="store_true",
        help="Treat hedging phrases as blocking errors (default: warning only)",
    )
    p.add_argument(
        "--allow-assumption-phrases",
        action="store_true",
        help="Do not fail on 'we assume' / unverified-assumption style phrases",
    )
    p.add_argument(
        "--allow-uncited-claims",
        action="store_true",
        help="Disable strong-claim-without-citation detection",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write full IntegrityReport JSON to this path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from governance.integrity_agent import IntegrityAgent

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text or ""

    agent = IntegrityAgent(
        fail_on_hedging=args.fail_on_hedging,
        fail_on_assumption_phrases=not args.allow_assumption_phrases,
        fail_on_uncited_strong_claims=not args.allow_uncited_claims,
    )
    chunks = _load_chunks(args.chunks_file)
    report = agent.scan(text, chunks=chunks)

    for v in report.violations:
        log = logger.error if v.severity == "error" else logger.warning
        log("[%s] %s — %s", v.rule_id, v.message, v.excerpt)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(
                {
                    "passed": report.passed,
                    "stats": report.stats,
                    "violations": [
                        {
                            "rule_id": v.rule_id,
                            "severity": v.severity,
                            "message": v.message,
                            "excerpt": v.excerpt,
                        }
                        for v in report.violations
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if report.passed:
        logger.info("Integrity gate PASSED")
        sys.exit(0)
    logger.error("Integrity gate FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
