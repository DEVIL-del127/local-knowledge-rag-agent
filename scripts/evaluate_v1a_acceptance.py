from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.v1a_acceptance import compare_candidate, load_jsonl, summarize_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and compare frozen V1-A acceptance evidence")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    corpus = load_jsonl(args.corpus)
    baseline = summarize_run(corpus, load_jsonl(args.baseline))
    candidate = summarize_run(corpus, load_jsonl(args.candidate))
    report = compare_candidate(baseline, candidate)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
