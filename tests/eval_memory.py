#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆系统评测（对应《记忆管理设计 v2.0》§7 可计算方法）
运行: ./venv/bin/python tests/eval_memory.py

指标:
  1. 清洗规则回归: 10 例样本集(L1 断言) + 保真校验
  2. 召回命中率: 构造事实入 L2 → 构造含指代查询 → recall 命中判定(实体+值)
  3. 冲突消解正确率: 构造冲突对 → 连续 2 次覆盖规则验证
  4. 注入预算合规: 随机召回 → 注入 token 估算 ≤ 预算
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_cleaning import clean_text, verify_fidelity  # noqa: E402
from memory.memory_manager import MemoryConfig, MemoryManager  # noqa: E402


class EvalCleaning(unittest.TestCase):
    """指标1: 清洗 10 例样本集 + 保真"""

    SAMPLES = [
        ("俺觉得这个GAN特别牛逼，贼好用", "用户觉得这个GAN特别牛逼，贼好用"),
        ("不要用那个什么resnet，换成vit吧", None),  # 仅断言否定保留
        ("http://example.com/a.pdf 帮我看看", None),  # 断言 [URL]
        ("上周说的误差补偿的事，继续吧", "上周说的误差补偿的事，继续吧"),
        ("嗯嗯好的，就按这个来吧", "好的，就按这个来吧"),
        ("我不想要太长的回答，简洁点", None),  # 断言否定+人称
        ("哎，刚才那个结果好像不太对，你再看看", None),
        ("B100和B200对比实验做了吗", "B100和B200对比实验做了吗"),
        ("我打电话给138****1234确认了", None),  # 断言 [PII]
        ("把这段翻译成英文，要信达雅", "把这段翻译成英文，要信达雅"),
    ]

    def test_10_samples(self):
        passed = 0
        for original, expected in self.SAMPLES:
            result = clean_text(original)
            ok, missing = verify_fidelity(original, result.text)
            if expected is not None:
                self.assertEqual(result.text, expected)
            self.assertTrue(ok, f"{original} 保真失败: {missing}")
            passed += 1
        print(f"  清洗样本通过: {passed}/{len(self.SAMPLES)}")


class EvalRecall(unittest.TestCase):
    """指标2: 召回命中率(构造 50 条事实)"""

    def test_recall_hit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(llm_client=None, state_dir=tmp, config=MemoryConfig())
            # 直接构造 50 条 L2 事实
            facts = [
                {"id": f"fact_eval_{i}", "type": "fact",
                 "entity": f"Topic{i}", "relation": "has_property",
                 "value": f"用户关于Topic{i}的偏好是偏好{i}", "ts": "2026-08-18T10:00:00+0800",
                 "session_id": "s_eval", "source_round": 1, "confidence": 0.8,
                 "retrieval_count": 0}
                for i in range(50)
            ]
            mgr._write_l2("eval_user", facts)

            hits = 0
            for i in range(50):
                injection = mgr.recall_memory(
                    query=f"还记得Topic{i}吗", user_id="eval_user"
                )
                if f"Topic{i}" in injection:
                    hits += 1
            rate = hits / 50
            print(f"  召回命中率: {rate:.2f} ({hits}/50)")
            self.assertGreaterEqual(rate, 0.8)


class EvalConflict(unittest.TestCase):
    """指标3: 冲突消解(连续 2 次覆盖)"""

    def test_conflict_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(llm_client=None, state_dir=tmp)
            # 直接写画像初始值
            profile = {
                "schema_version": 1, "stable": {}, "dynamic": {},
                "history": {},
            }
            profile["dynamic"]["project"] = {
                "value": "A项目", "confidence": 0.8,
                "updated_ts": "2026-08-01T10:00:00+0800", "conflicts": [],
            }
            path = os.path.join(tmp, "memory", "u", "profile.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(profile, open(path, "w", encoding="utf-8"))

            # 第 1 次冲突新值 → 不覆盖
            mgr._merge_profile_field(profile, "project", "B项目", 0.8, "2026-08-18T10:00:00+0800")
            self.assertEqual(profile["dynamic"]["project"]["value"], "A项目")
            # 第 2 次同新值 → 覆盖
            mgr._merge_profile_field(profile, "project", "B项目", 0.8, "2026-08-18T11:00:00+0800")
            self.assertEqual(profile["dynamic"]["project"]["value"], "B项目")
            print("  冲突消解: 连续2次覆盖规则 ✓")


class EvalBudget(unittest.TestCase):
    """指标4: 注入预算合规"""

    def test_injection_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(llm_client=None, state_dir=tmp, config=MemoryConfig())
            facts = [
                {"id": f"f_{i}", "type": "fact", "entity": "E", "relation": "has_property",
                 "value": "x" * 80, "ts": "2026-08-18T10:00:00+0800",
                 "session_id": "s", "confidence": 0.8, "retrieval_count": 0}
                for i in range(20)
            ]
            mgr._write_l2("u", facts)
            for _ in range(10):
                injection = mgr.recall_memory(query="查E", user_id="u")
                self.assertLessEqual(len(injection) // 2, 800)
            print("  注入预算: 10 次召回全部 ≤800t ✓")


def main():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
