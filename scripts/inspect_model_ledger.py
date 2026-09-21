"""Print aggregate model ledger metadata without persisted request/response content."""
import argparse
import json
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
args = parser.parse_args()
with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    result = {"tables": sorted(tables.intersection({
        "model_calls", "token_budget_buckets", "token_budget_reservations"
    }))}
    if "model_calls" in tables:
        values = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM model_calls")]
        result["states"] = dict(connection.execute(
            "SELECT state,COUNT(*) FROM model_calls GROUP BY state"
        ).fetchall())
        result["purposes"] = {
            name: sum(1 for value in values if value.get("purpose") == name)
            for name in sorted({str(value.get("purpose") or "") for value in values})
        }
        result["errors"] = {
            name: sum(1 for value in values if value.get("error_class") == name)
            for name in sorted({str(value.get("error_class") or "") for value in values
                                if value.get("error_class")})
        }
        result["attempt_count"] = len(values)
    print(json.dumps(result, sort_keys=True))
