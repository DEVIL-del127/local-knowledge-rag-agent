# -*- coding: utf-8 -*-
"""nlu_validate 标注功能专项测试：窗口解析 / 标注落盘 / 统计聚合

纯函数部分（_parse_window / _save_annotation / _load_annotations / stats）
不依赖模型/ES，秒开。
"""
import sys
import os
import json
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
from nlu_validate import _parse_window, _save_annotation, _load_annotations, _annotations_stats

# ---------- 时间窗口解析矩阵 ----------
WINDOW_CASES = [
    # (输入, 期望 (n, unit) 或 None)
    ("6个月", (6, "month")),
    ("3年", (3, "year")),
    ("1周", (1, "week")),
    ("2天", (2, "day")),
    ("半年", (6, "month")),
    ("一年", (1, "year")),
    ("一个月", (1, "month")),
    ("两周", (2, "week")),
    ("三个月", (3, "month")),
    ("", None),
    ("  ", None),
    ("abc", None),
    ("大概6个月", (6, "month")),
]

# ---------- 标注记录结构（字段完整性） ----------
REQUIRED_FIELDS = ["ts", "query", "cleaned", "intent", "source", "label"]


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return path


def main():
    fails = []
    ok = 0

    # 1. 窗口解析
    for inp, exp in WINDOW_CASES:
        got = _parse_window(inp)
        if got == exp:
            ok += 1
        else:
            fails.append(("WINDOW", inp, exp, got))

    # 2. 落盘 + 回读 roundtrip
    path = _tmp_path()
    rec = {
        "ts": "2026-08-21T14:47:00+08:00",
        "query": "找最近一年的文献",
        "cleaned": "找最近一年的文献",
        "intent": "attribute_filter",
        "source": "rule_fuzzy_default",
        "label": "wrong",
        "expected_intent": "inventory",
        "note": "应该是统计",
        "time": {"op": "gte", "from": "2026-05-01", "granularity": "month", "fuzzy_word": "最近"},
        "time_expected": {"fuzzy_word": "最近", "window": 12, "unit": "month"},
        "entities": {"author": None, "venue": None, "language": None, "doc_type": None, "doc_ids": [], "exclude": {}},
    }
    try:
        _save_annotation(rec, path)
        recs = _load_annotations(path)
        if len(recs) == 1 and all(k in recs[0] for k in REQUIRED_FIELDS) \
                and recs[0]["label"] == "wrong" and recs[0]["time_expected"]["window"] == 12:
            ok += 1
        else:
            fails.append(("ROUNDTRIP", path, "单条完整回读", recs))
        # 追加不覆盖
        rec2 = dict(rec)
        rec2["label"] = "correct"
        _save_annotation(rec2, path)
        recs = _load_annotations(path)
        if len(recs) == 2:
            ok += 1
        else:
            fails.append(("APPEND", path, "追加第二条", len(recs)))
        # 空文件容错
        if _load_annotations(path + ".nonexist") == []:
            ok += 1
        else:
            fails.append(("EMPTY", path, "不存在文件→[]", _load_annotations(path + ".nonexist")))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    # 3. 统计聚合
    path2 = _tmp_path()
    try:
        for label in ["correct", "correct", "wrong", "skip", "wrong"]:
            _save_annotation({"ts": "t", "query": "q", "cleaned": "q",
                              "intent": "attribute_filter" if label != "wrong" else "inventory",
                              "source": "rule_x", "label": label}, path2)
        st = _annotations_stats(path2)
        if st["total"] == 5 and st["labels"]["correct"] == 2 \
                and st["labels"]["wrong"] == 2 and st["labels"]["skip"] == 1 \
                and st["accuracy"] == 0.5:
            ok += 1
        else:
            fails.append(("STATS", path2, "5条统计", st))
        # 空库统计容错
        st0 = _annotations_stats(path2 + ".nonexist")
        if st0["total"] == 0:
            ok += 1
        else:
            fails.append(("STATS_EMPTY", path2, "空库", st0))
    finally:
        try:
            os.remove(path2)
        except OSError:
            pass

    total = len(WINDOW_CASES) + 5
    print(f"标注功能测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] {q}\n     期望={e}\n     实际={g}")
        sys.exit(1)
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
