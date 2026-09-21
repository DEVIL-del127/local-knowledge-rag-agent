"""Read-only counts and formats; never print stored values or key bytes."""
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from agent.app_settings import AppSettings


def main():
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env(ROOT)
    state = settings.state_dir
    result = {"state_directory_exists": state.is_dir(), "databases": [], "legacy_key_file_count": 0}
    if state.is_dir():
        result["legacy_key_file_count"] = sum(1 for p in state.rglob("*.key") if p.is_file() and p.stat().st_size == 32)
    for relative, table, column in (
        ("checkpoints/runtime.sqlite3", "checkpoints", "payload"),
        ("model_calls.sqlite3", "model_calls", "payload"),
    ):
        path = state / relative
        item = {"role": relative, "exists": path.is_file()}
        if path.is_file():
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if exists:
                    item["rows"] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    if table == "checkpoints":
                        item["non_enveloped_rows"] = connection.execute(
                            "SELECT COUNT(*) FROM checkpoints WHERE payload NOT LIKE 'enc-v1:%'"
                        ).fetchone()[0]
            finally:
                connection.close()
        result["databases"].append(item)
    usage = state / "token_usage.json"
    result["legacy_usage_file_exists"] = usage.is_file()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
