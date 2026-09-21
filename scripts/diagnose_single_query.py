"""Operator-only single-query traceback reproducer; never prints answers."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from agent.app_settings import AppSettings
from agent.application import build_application

parser = argparse.ArgumentParser()
parser.add_argument("query")
args = parser.parse_args()
if os.environ.get("AUTHORIZE_PRIVATE_DEEPSEEK_E2E") != "1":
    raise SystemExit("authorization required")
load_dotenv(ROOT / ".env")
app = build_application(AppSettings.from_env(ROOT))
try:
    reply = app.agent.chat(args.query, user_id="deepseek-diagnostic",
                           session_id="single-diagnostic", thread_id="single-diagnostic", history=[])
    print({"intent": str(reply.intent.intent), "evidence_count": len(reply.evidence),
           "answer_nonempty": bool(reply.answer.strip())})
finally:
    app.close()
