"""批量测试脚本：将测试.txt中的所有题目逐个送入nlu_v2_validate.py测试，结果保存为JSON文件到test文件夹。"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "测试.txt"
OUTPUT_DIR = BASE_DIR / "test"
VALIDATE_SCRIPT = BASE_DIR / "nlu_v2_validate.py"
PYTHON_EXEC = sys.executable or "python3"


def parse_test_cases(content: str) -> list[dict]:
    """解析测试.txt，返回 [{id, title, query}, ...]"""
    pattern = re.compile(
        r"###\s*测试题\s*(\d+)[：:]\s*([^\n]+)\s*\n\s*>\s*([\s\S]*?)(?=\n------|\n###\s*测试题|$)",
        re.MULTILINE,
    )
    cases = []
    for match in pattern.finditer(content):
        qid = int(match.group(1))
        title = match.group(2).strip()
        query = match.group(3).strip()
        query = re.sub(r"^\s*>\s*", "", query, flags=re.MULTILINE)
        query = query.replace("\n", " ").strip()
        cases.append({"id": qid, "title": title, "query": query})
    cases.sort(key=lambda x: x["id"])
    return cases


def run_single_case(query: str, use_llm: bool = True,
                    llm_model: str | None = None,
                    candidate_choice_v3: bool = False,
                    m15_shadow: bool = False) -> str:
    """调用 nlu_v2_validate.py 执行单条测试，返回 JSON 字符串（原始输出）。"""
    cmd = [PYTHON_EXEC, str(VALIDATE_SCRIPT), query, "--json"]
    if not use_llm:
        cmd.append("--no-llm")
    elif llm_model:
        cmd.extend(["--llm-model", llm_model])
    if candidate_choice_v3:
        cmd.append("--candidate-choice-v3")
    if m15_shadow:
        cmd.append("--m15-shadow")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(BASE_DIR),
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        error_obj = {
            "status": "error",
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        return json.dumps(error_obj, ensure_ascii=False, indent=2)
    if not stdout:
        return json.dumps({"status": "empty", "stderr": stderr}, ensure_ascii=False, indent=2)
    return stdout


def safe_filename(qid: int, title: str) -> str:
    safe_title = re.sub(r"[^\w\u4e00-\u9fff-]", "_", title).strip("_")
    if len(safe_title) > 60:
        safe_title = safe_title[:60]
    return f"case_{qid:03d}_{safe_title}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量运行冻结 NLU V2 测试题")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="结果目录；默认仍为 kb-agent/test",
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--candidate-choice-v3", action="store_true")
    parser.add_argument("--m15-shadow", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if not TEST_FILE.exists():
        print(f"[ERROR] 未找到测试文件: {TEST_FILE}", file=sys.stderr)
        return 1
    if not VALIDATE_SCRIPT.exists():
        print(f"[ERROR] 未找到验证脚本: {VALIDATE_SCRIPT}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    content = TEST_FILE.read_text(encoding="utf-8")
    cases = parse_test_cases(content)
    total = len(cases)
    print(f"[INFO] 共解析到 {total} 道测试题，输出目录: {output_dir}")

    summary = []
    for idx, case in enumerate(cases, 1):
        qid = case["id"]
        title = case["title"]
        query = case["query"]
        print(f"[{idx}/{total}] 正在测试 题{qid}: {title} ...", end=" ", flush=True)

        json_output = run_single_case(
            query, use_llm=not args.no_llm, llm_model=args.llm_model,
            candidate_choice_v3=args.candidate_choice_v3,
            m15_shadow=args.m15_shadow,
        )

        filename = safe_filename(qid, title)
        filepath = output_dir / filename
        filepath.write_text(json_output, encoding="utf-8")

        try:
            parsed = json.loads(json_output)
            status = parsed.get("status", "unknown")
            if isinstance(parsed, dict) and "validation" in parsed:
                val = parsed["validation"]
                status = f"valid:{val.get('status','?')} exe:{val.get('executable','?')}"
        except (json.JSONDecodeError, TypeError):
            status = "invalid_json"

        print(f"-> {status}  [已保存 {filename}]")
        summary.append({
            "id": qid,
            "title": title,
            "file": filename,
            "status": status,
        })

    summary_path = output_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[DONE] 全部 {total} 道题已完成测试。汇总文件: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
