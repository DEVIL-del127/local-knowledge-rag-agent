from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect generation registry and stale builds")
    parser.add_argument("registry", nargs="?", default="data/ingestion/generation_registry.sqlite3")
    parser.add_argument(
        "--fail-orphans", action="store_true",
        help="mark BUILDING/VALIDATING records without a manifest as FAILED",
    )
    args = parser.parse_args()
    path = Path(args.registry)
    if not path.exists():
        print(json.dumps({"registry": str(path), "exists": False}, ensure_ascii=False))
        return 2
    connection = sqlite3.connect(str(path))
    try:
        meta = connection.execute(
            "SELECT active_generation, revision FROM registry_meta WHERE singleton=1"
        ).fetchone()
        rows = connection.execute(
            "SELECT generation_id, payload FROM generations ORDER BY generation_id"
        ).fetchall()
    finally:
        connection.close()
    generations = []
    for generation_id, raw in rows:
        payload = json.loads(raw)
        payload["generation_id"] = generation_id
        payload["orphan_build"] = (
            payload.get("state") in {"building", "validating"}
            and not payload.get("manifest_locator")
        )
        generations.append(payload)
    if args.fail_orphans:
        from ingestion.pipeline.generation_registry import (
            GenerationRegistry, GenerationState,
        )
        registry = GenerationRegistry(path)
        for payload in generations:
            if not payload["orphan_build"]:
                continue
            record = registry.get(payload["generation_id"])
            if record is not None:
                record.state = GenerationState.FAILED
                record.failure_reason = "startup recovery: stale build has no immutable manifest"
                registry.put(record)
                payload["state"] = "failed"
                payload["failure_reason"] = record.failure_reason
                payload["orphan_build"] = False
    print(json.dumps({
        "registry": str(path.resolve()), "exists": True,
        "active_generation": meta[0] if meta else None,
        "revision": int(meta[1]) if meta else 0,
        "generations": generations,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
