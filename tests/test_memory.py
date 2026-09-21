#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆系统 M1 单元测试
运行: ./venv/bin/python tests/test_memory.py
覆盖: 清洗 10 例样本集 / 保真校验 / 字面去重 / 锚词冲突 / 会话生命周期 / 崩溃恢复 / 提取降级
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_cleaning import (  # noqa: E402
    anchor_conflict,
    clean_text,
    literal_dedup,
    verify_fidelity,
)
from memory.memory_manager import MemoryConfig, MemoryManager  # noqa: E402
from memory.memory_retrieval import (  # noqa: E402
    detect_anaphora,
    extract_entities,
    format_injection,
    keyword_search,
    rewrite_query_with_context,
)
from memory.memory_skills import MemoryManageSkill, MemoryRecallSkill  # noqa: E402


class TestCleaningSamples(unittest.TestCase):
    """§2.2.2 清洗 10 例样本集(L1 层)"""

    def test_sample1_colloquial(self):
        r = clean_text("俺觉得这个GAN特别牛逼，贼好用")
        self.assertEqual(r.text, "用户觉得这个GAN特别牛逼，贼好用")

    def test_sample2_negation_preserved(self):
        r = clean_text("不要用那个什么resnet，换成vit吧")
        self.assertIn("不要", r.text)
        self.assertTrue(r.negation_spans)

    def test_sample3_url_masked(self):
        r = clean_text("http://example.com/a.pdf 帮我看看")
        self.assertIn("[URL]", r.text)
        self.assertNotIn("http://", r.text)

    def test_sample4_time_kept(self):
        r = clean_text("上周说的误差补偿的事，继续吧")
        self.assertEqual(r.text, "上周说的误差补偿的事，继续吧")

    def test_sample5_filler_removed(self):
        r = clean_text("嗯嗯好的，就按这个来吧")
        self.assertEqual(r.text, "好的，就按这个来吧")

    def test_sample6_negation_frozen(self):
        r = clean_text("我不想要太长的回答，简洁点")
        self.assertIn("不想要", r.text)
        self.assertIn("用户", r.text)  # 人称替换

    def test_sample7_correction_kept(self):
        r = clean_text("哎，刚才那个结果好像不太对，你再看看")
        self.assertNotIn("哎", r.text)
        self.assertIn("Agent", r.text)

    def test_sample8_numbers_kept(self):
        r = clean_text("B100和B200对比实验做了吗")
        self.assertIn("B100", r.text)
        self.assertIn("B200", r.text)

    def test_sample9_pii_masked(self):
        r = clean_text("我打电话给138****1234确认了")
        self.assertIn("[PII]", r.text)
        self.assertNotIn("138", r.text)

    def test_sample10_constraint_kept(self):
        r = clean_text("把这段翻译成英文，要信达雅")
        self.assertEqual(r.text, "把这段翻译成英文，要信达雅")


class TestFidelity(unittest.TestCase):
    def test_numbers_missing_detected(self):
        ok, missing = verify_fidelity("对比实验B100", "对比实验")
        self.assertFalse(ok)
        self.assertTrue(any("100" in m for m in missing), f"missing={missing}")

    def test_negation_missing_detected(self):
        ok, missing = verify_fidelity("用户不想要X", "用户想要X")
        self.assertFalse(ok)

    def test_clean_pass(self):
        ok, _ = verify_fidelity("B100实验做了", clean_text("B100实验做了").text)
        self.assertTrue(ok)


class TestDedupConflict(unittest.TestCase):
    def test_literal_dedup(self):
        existing = [{"id": "f1", "content": "用户偏好简洁回答"}]
        new = [{"content": "用户偏好简洁回答"}, {"content": "用户是研究生"}]
        result = literal_dedup(new, existing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "用户是研究生")

    def test_anchor_conflict(self):
        existing = [{"id": "f1", "type": "preference", "content": "用户偏好ResNet模型"}]
        new = [{"type": "preference", "content": "用户改为用ViT模型, 不要ResNet"}]
        result = anchor_conflict(new, existing)
        self.assertEqual(result[0].get("conflict_with"), "f1")


class TestSessionLifecycle(unittest.TestCase):
    def _manager(self, tmp):
        return MemoryManager(llm_client=None, state_dir=tmp, config=MemoryConfig(extract_every_n_rounds=1))

    def test_session_roundtrip_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            sid = mgr.on_session_start(user_id="u1")
            self.assertFalse(os.path.exists(os.path.join(tmp, "sessions", f"{sid}.json")))

            mgr.ingest_message(round_no=1, user_msg="我偏好简洁回答",
                               agent_reply="好的", intent=None, session_id=sid, user_id="u1")
            mgr.on_session_end(session_id=sid, user_id="u1")

            # 会话归档
            session = mgr._load_archived_session(sid)
            self.assertIsNotNone(session["ended_ts"])
            self.assertEqual(session["end_reason"], "normal")
            self.assertEqual(session["facts"], [])

    def test_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            sid = mgr.on_session_start(user_id="u1")
            mgr.ingest_message(round_no=1, user_msg="用户偏好简洁回答",
                               agent_reply="好的", intent=None, session_id=sid, user_id="u1")
            # 模拟崩溃: 不调 on_session_end, 直接改 updated_ts 为 2 天前
            path = os.path.join(tmp, "sessions", f"{sid}.json")
            session = mgr._load_session(sid)
            session["updated_ts"] = "2026-08-16T10:00:00+0800"
            mgr._save_session(session)

            # 普通启动不得隐式迁移、晋级或归档旧会话。
            mgr.on_session_start(user_id="u1")
            self.assertIsNotNone(mgr._load_session(sid))
            self.assertIsNone(mgr._load_archived_session(sid))

    def test_raw_fallback_without_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            sid = mgr.on_session_start(user_id="u1")
            mgr.ingest_message(round_no=1, user_msg="B100对比实验做完了",
                               agent_reply="好的", intent=None, session_id=sid, user_id="u1")
            session = mgr._load_session(sid)
            self.assertEqual(session["facts"], [])
            mgr.on_session_end(session_id=sid, user_id="u1")


class TestProfile(unittest.TestCase):
    def test_profile_merge_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(llm_client=None, state_dir=tmp)
            sid = mgr.on_session_start(user_id="u2")
            # 非显式候选不得在退出时更新画像。
            session = mgr._load_session(sid)
            session["profile_candidates"] = [
                {"field": "project", "value": "RAG系统", "confidence": 0.8, "ts": "2026-08-18T10:00:00+0800"},
            ]
            mgr._save_session(session)
            mgr.on_session_end(session_id=sid, user_id="u2")

            profile = mgr._load_profile("u2")
            self.assertNotIn("project", profile["dynamic"])

            # 重复注入仍不构成授权。
            sid2 = mgr.on_session_start(user_id="u2")
            session2 = mgr._load_session(sid2)
            session2["profile_candidates"] = [
                {"field": "project", "value": "写作助手", "confidence": 0.8, "ts": "2026-08-18T11:00:00+0800"},
            ]
            mgr._save_session(session2)
            mgr.on_session_end(session_id=sid2, user_id="u2")
            profile = mgr._load_profile("u2")
            self.assertNotIn("project", profile["dynamic"])

            sid3 = mgr.on_session_start(user_id="u2")
            session3 = mgr._load_session(sid3)
            session3["profile_candidates"] = [
                {"field": "project", "value": "写作助手", "confidence": 0.8, "ts": "2026-08-18T12:00:00+0800"},
            ]
            mgr._save_session(session3)
            mgr.on_session_end(session_id=sid3, user_id="u2")
            profile = mgr._load_profile("u2")
            self.assertNotIn("project", profile["dynamic"])


class TestAnaphora(unittest.TestCase):
    def test_detect_without_antecedent(self):
        # 含指代且窗口内无实体 → 触发
        self.assertTrue(detect_anaphora("那个结果怎么样了", []))
        self.assertTrue(detect_anaphora("它完成了吗", ["好的，继续", "嗯嗯"]))

    def test_detect_with_antecedent(self):
        # 窗口内有实体词 → 不触发(指代可在上下文解析)
        self.assertFalse(detect_anaphora("那个结果怎么样了", ["B100对比实验做完了"]))

    def test_long_message_no_detect(self):
        self.assertFalse(detect_anaphora("那个结果怎么样了" + "的详细分析报告已经出来了" * 10, []))

    def test_phrase_anchor(self):
        self.assertTrue(detect_anaphora("上次说的GAN那个事", []))


class TestEntityExtract(unittest.TestCase):
    def test_english_and_chinese(self):
        entities = extract_entities("GAN模型和B100实验")
        self.assertIn("GAN", entities)
        self.assertIn("B100", entities)

    def test_stopwords_excluded(self):
        entities = extract_entities("那个什么怎么样")
        self.assertEqual(entities, [])


class TestKeywordSearch(unittest.TestCase):
    def test_match_by_entity(self):
        facts = [
            {"content": "用户偏好ResNet模型", "confidence": 0.8},
            {"content": "用户写了网文大纲", "confidence": 0.9},
        ]
        result = keyword_search(facts, ["ResNet"], top_k=5)
        self.assertEqual(len(result), 1)
        self.assertIn("ResNet", result[0]["content"])


class TestFormatInjection(unittest.TestCase):
    def test_session_format(self):
        items = [{"type": "preference", "content": "用户偏好简洁回答", "confidence": 0.8}]
        text = format_injection("会话记忆", items)
        self.assertIn("[会话记忆]", text)
        self.assertIn("偏好简洁回答", text)

    def test_budget_cut(self):
        items = [{"type": "fact", "content": "x" * 200} for _ in range(10)]
        text = format_injection("会话记忆", items, budget_tokens=30)
        self.assertLessEqual(len(text) // 2, 30)

    def test_empty(self):
        self.assertEqual(format_injection("会话记忆", []), "")


class TestRecallAndAdmin(unittest.TestCase):
    def _mgr(self, tmp):
        return MemoryManager(llm_client=None, state_dir=tmp, config=MemoryConfig(extract_every_n_rounds=1))

    def test_recall_for_current_anaphora(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._mgr(tmp)
            sid = mgr.on_session_start(user_id="u1")
            mgr.ingest_message(round_no=1, user_msg="我偏好ResNet模型",
                               agent_reply="好的", intent=None, session_id=sid, user_id="u1")
            # 含指代查询
            injection = mgr.recall_for_current(
                query="那个模型怎么样", window_messages=[],
                session_id=sid, user_id="u1",
            )
            self.assertIsNone(injection)
            # 无指代查询 → 不注入
            injection2 = mgr.recall_for_current(
                query="今天天气怎么样", window_messages=["普通消息"],
                session_id=sid, user_id="u1",
            )
            self.assertIsNone(injection2)
            mgr.on_session_end(session_id=sid, user_id="u1")

    def test_recall_memory_cross_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._mgr(tmp)
            # 直接构造 L2 数据(跳过 LLM 提取, 测召回逻辑)
            mgr._write_l2("u1", [
                {"id": "fact_x", "type": "fact", "entity": "GAN",
                 "relation": "has_property", "value": "用户偏好GAN生成对抗网络",
                 "ts": "2026-08-18T10:00:00+0800", "session_id": "s1",
                 "confidence": 0.8, "retrieval_count": 0},
            ])
            injection = mgr.recall_memory(query="还记得我上次说的GAN吗", user_id="u1")
            self.assertIn("GAN", injection)

    def test_admin_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._mgr(tmp)
            mgr._write_l2("u1", [
                {"id": "fact_a", "type": "fact", "entity": "E1",
                 "relation": "has_property", "value": "用户偏好简洁回答",
                 "ts": "2026-08-18T10:00:00+0800", "session_id": "s1",
                 "confidence": 0.8, "retrieval_count": 0},
                {"id": "fact_b", "type": "fact", "entity": "E2",
                 "relation": "has_property", "value": "用户写了网文大纲",
                 "ts": "2026-08-18T11:00:00+0800", "session_id": "s1",
                 "confidence": 0.9, "retrieval_count": 0},
            ])

            stats = mgr.admin(command="stats", user_id="u1")
            self.assertEqual(stats["total"], 2)

            listed = mgr.admin(command="list", user_id="u1", args={"limit": 5})
            self.assertGreaterEqual(len(listed["items"]), 1)
            target_id = listed["items"][0]["id"]

            deleted = mgr.admin(command="delete", user_id="u1", args={"id": target_id})
            self.assertEqual(deleted["deleted"], target_id)
            self.assertIn("error", mgr.admin(command="get", user_id="u1", args={"id": target_id}))

            exported = mgr.admin(command="export", user_id="u1")
            self.assertIsInstance(exported["exported"], list)

            cleared = mgr.admin(command="clear", user_id="u1")
            self.assertTrue(cleared["cleared"])
            self.assertEqual(mgr.admin(command="stats", user_id="u1")["total"], 0)

    def test_admin_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._mgr(tmp)
            mgr._write_l2("u1", [
                {"id": "fact_a", "type": "fact", "entity": "E1",
                 "relation": "has_property", "value": "用户偏好简洁回答",
                 "ts": "2026-08-18T10:00:00+0800", "session_id": "s1",
                 "confidence": 0.8, "retrieval_count": 0},
            ])
            listed = mgr.admin(command="list", user_id="u1", args={"limit": 5})
            target_id = listed["items"][0]["id"]
            fixed = mgr.admin(command="fix", user_id="u1",
                              args={"id": target_id, "value": "用户偏好详细回答"})
            self.assertEqual(fixed["fixed"], target_id)
            got = mgr.admin(command="get", user_id="u1", args={"id": target_id})
            self.assertEqual(got["item"]["value"], "用户偏好详细回答")

    def test_skills_invoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._mgr(tmp)
            sid = mgr.on_session_start(user_id="u1")
            mgr.ingest_message(round_no=1, user_msg="我偏好ResNet模型",
                               agent_reply="好", intent=None, session_id=sid, user_id="u1")

            recall_skill = MemoryRecallSkill(mgr)
            result = recall_skill.invoke({"query": "那个", "scope": "current",
                                          "user_id": "u1", "session_id": sid})
            self.assertFalse(result["found"])

            manage_skill = MemoryManageSkill(mgr)
            stats = manage_skill.invoke({"command": "stats", "user_id": "u1"})
            self.assertIn("total", stats)
            mgr.on_session_end(session_id=sid, user_id="u1")


class TestQueryRewrite(unittest.TestCase):
    """多轮指代查询重写"""

    def test_anaphora_subject_omission(self):
        # 用户场景: "刘月的论文写了什么" → "用了哪些技术和模型"
        rewritten = rewrite_query_with_context(
            "用了哪些技术和模型", ["刘月的论文写了什么"]
        )
        self.assertIn("刘月的论文", rewritten)
        self.assertIn("技术", rewritten)

    def test_has_entity_no_rewrite(self):
        rewritten = rewrite_query_with_context(
            "刘月论文用了什么技术", ["刘月的论文写了什么"]
        )
        self.assertEqual(rewritten, "刘月论文用了什么技术")

    def test_no_window_entity_no_rewrite(self):
        rewritten = rewrite_query_with_context("用了哪些技术", ["今天天气不错"])
        self.assertEqual(rewritten, "用了哪些技术")

    def test_english_entity(self):
        rewritten = rewrite_query_with_context("它是什么原理", ["GAN模型讲一下"])
        self.assertIn("GAN", rewritten)

    def test_entity_extract_excludes_question_tokens(self):
        entities = extract_entities("刘月的论文写了什么")
        self.assertIn("刘月的论文", entities)
        self.assertNotIn("写了什么", entities)

    def test_deictic_phrase_replace(self):
        # "上次查的论文" 是指示短语, 应替换为窗口实体
        rewritten = rewrite_query_with_context(
            "上次查的论文用了哪些模型技术", ["刘月的论文写了什么"]
        )
        self.assertIn("刘月的论文", rewritten)
        self.assertNotIn("上次", rewritten)
        self.assertIn("模型技术", rewritten)

    def test_deictic_suffix_clean(self):
        # "刘月那篇" 尾部指示后缀应清理
        rewritten = rewrite_query_with_context("刘月那篇", ["刘月的论文写了什么"])
        self.assertEqual(rewritten, "刘月")

    def test_deictic_variants(self):
        w = ["刘月的论文写了什么"]
        self.assertIn("刘月的论文", rewrite_query_with_context("上次那个方案怎么样了", w))
        self.assertIn("刘月的论文", rewrite_query_with_context("刚才说的模型再讲讲", w))
        self.assertNotIn("论文论文", rewrite_query_with_context("之前那篇论文讲了什么", w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
