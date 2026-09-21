"""Read-only aggregate inspection of one persisted E2E run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from agent.runtime.checkpoint_store import SQLiteCheckpointStore

parser = argparse.ArgumentParser()
parser.add_argument("--database", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--total", type=int, default=80)
args = parser.parse_args()
load_dotenv(ROOT / ".env")
store = SQLiteCheckpointStore(args.database)
rows = []
for index in range(1, args.total + 1):
    user = f"release-v31-{args.run_id}"
    scope = f"{args.run_id}-e2e-{index:03d}"
    state = store.load(user, scope, scope)
    rows.append({
        "index": index,
        "state": getattr(getattr(state, "state", None), "value", "missing"),
        "reason": str(getattr(state, "reason", "") or ""),
        "evidence_count": len(getattr(state, "evidence", []) or []) if state else 0,
    })
print(json.dumps({"complete": sum(row["state"] == "complete" for row in rows),
                  "non_complete": [row for row in rows if row["state"] != "complete"]},
                 ensure_ascii=False, sort_keys=True))
