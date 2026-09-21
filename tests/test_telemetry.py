# -*- coding: utf-8 -*-
"""日志系统(问题侧 only): TelemetryBus 组装 / 保留策略 / ForwardingStorage 断点续传"""
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry.bus import TelemetryBus
from telemetry.context import RequestContext
from telemetry.search_log.config import SearchLogConfig
from telemetry.search_log.models import SearchLogEvent
from telemetry.search_log.recorder import SearchRecorder
from telemetry.search_log.storage import (
    ForwardingStorage,
    JsonFileStorage,
    SqliteStorage,
)


def _make_ctx(**overrides) -> RequestContext:
    ctx = RequestContext(
        request_id="req_test123",
        user_id="u_42",
        session_id="sess_1",
        node_id="node-a",
        query_raw="刘月的论文用了什么模型？？顺便讲ESN",
        query_cleaned="刘月的论文用了什么模型?顺便讲ESN",
        query_primary="刘月的论文用了什么模型",
        query_subs=["顺便讲ESN"],
        query_rewritten="刘月的论文 用了什么模型",
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


class TestTelemetryBus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = SearchLogConfig(
            json_dir=self.tmp, hash_salt="salt",
            queue_size=100, batch_size=10, flush_interval=0.1,
        )

    def _read_lines(self, prefix: str) -> list[dict]:
        day = time.strftime("%Y-%m-%d")
        path = os.path.join(self.tmp, f"{prefix}_{day}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    def test_emit_writes_question_only(self):
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx())
        bus.shutdown()
        search = self._read_lines("search_log")
        self.assertEqual(len(search), 1)
        # 只落 search_log, 不落 trace
        self.assertEqual(self._read_lines("trace"), [])

    def test_question_fields_recorded(self):
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx())
        bus.shutdown()
        rec = self._read_lines("search_log")[0]
        self.assertEqual(rec["query_raw"], "刘月的论文用了什么模型？？顺便讲ESN")
        self.assertEqual(rec["query_cleaned"], "刘月的论文用了什么模型?顺便讲ESN")
        self.assertEqual(rec["query_primary"], "刘月的论文用了什么模型")
        self.assertEqual(rec["query_subs"], ["顺便讲ESN"])
        self.assertEqual(rec["query_rewritten"], "刘月的论文 用了什么模型")
        self.assertEqual(rec["node_id"], "node-a")
        self.assertEqual(rec["session_id"], "sess_1")
        self.assertEqual(rec["request_id"], "req_test123")

    def test_no_result_fields(self):
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx())
        bus.shutdown()
        rec = self._read_lines("search_log")[0]
        for field in ("intent", "total_hits", "evidence_count",
                      "latency_ms", "answer_len", "skill_steps"):
            self.assertNotIn(field, rec, f"不应记录 {field}")

    def test_user_id_hashed_with_salt(self):
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx())
        bus.shutdown()
        rec = self._read_lines("search_log")[0]
        self.assertNotEqual(rec["user_id"], "u_42")
        self.assertEqual(len(rec["user_id"]), 16)

    def test_user_id_plain_without_salt(self):
        self.config.hash_salt = ""
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx())
        bus.shutdown()
        rec = self._read_lines("search_log")[0]
        self.assertEqual(rec["user_id"], "u_42")

    def test_emit_idempotent(self):
        bus = TelemetryBus(node_id="node-a", config=self.config)
        ctx = _make_ctx()
        bus.emit(ctx)
        bus.emit(ctx)  # 同 request_id 只发一次
        bus.shutdown()
        self.assertEqual(len(self._read_lines("search_log")), 1)

    def test_fail_silent_on_bad_storage(self):
        class BoomStorage:
            def write(self, records, timeout_s=2.0):
                raise OSError("disk full")

        bus = TelemetryBus(node_id="node-a", config=self.config, storage=BoomStorage())
        bus.emit(_make_ctx())  # 不抛异常
        bus.shutdown()
        self.assertTrue(True)


class TestJsonFileStorageRetention(unittest.TestCase):
    def test_cleanup_removes_old_files(self):
        tmp = tempfile.mkdtemp()
        old = os.path.join(tmp, "search_log_2020-01-01.jsonl")
        with open(old, "w", encoding="utf-8") as f:
            f.write("{}\n")
        os.utime(old, (time.time() - 200 * 86400, time.time() - 200 * 86400))
        storage = JsonFileStorage(tmp, prefix="search_log", retention_days=90)
        storage.close()
        self.assertFalse(os.path.exists(old))

    def test_cleanup_keeps_recent_files(self):
        tmp = tempfile.mkdtemp()
        recent = os.path.join(tmp, "search_log_2026-01-01.jsonl")
        with open(recent, "w", encoding="utf-8") as f:
            f.write("{}\n")
        storage = JsonFileStorage(tmp, prefix="search_log", retention_days=90)
        storage.close()
        self.assertTrue(os.path.exists(recent))

    def test_retention_zero_disabled(self):
        tmp = tempfile.mkdtemp()
        old = os.path.join(tmp, "search_log_2020-01-01.jsonl")
        with open(old, "w", encoding="utf-8") as f:
            f.write("{}\n")
        os.utime(old, (time.time() - 500 * 86400, time.time() - 500 * 86400))
        storage = JsonFileStorage(tmp, prefix="search_log", retention_days=0)
        storage.close()
        self.assertTrue(os.path.exists(old))


class _MockCenter:
    """内存 mock 中心: 记录收到的批次, 可注入失败"""

    def __init__(self):
        self.received: list[list[dict]] = []
        self.node_ids: list[str] = []
        self.fail_next = False
        self._server = None
        self._thread = None
        self.port = 0

    def start(self):
        handler = _HandlerFactory(self)
        self._server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()


class _HandlerFactory:
    def __init__(self, center):
        self.center = center

    def __call__(self, *args, **kwargs):
        return _BatchHandler(self.center, *args, **kwargs)


class _BatchHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, center, *args, **kwargs):
        self.center = center
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.center.fail_next:
            self.center.fail_next = False
            self.send_response(500)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.center.received.append(body["events"])
        self.center.node_ids.append(body.get("node_id", ""))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestForwardingStorage(unittest.TestCase):
    def test_forward_once_posts_lines(self):
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
            )
            rec_path = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write('{"event_id":"a:0001","q":1}\n')
                f.write('{"event_id":"a:0002","q":2}\n')
            sent = fs._forward_once()
            self.assertEqual(sent, 2)
            self.assertEqual(len(center.received[0]), 2)
            sent = fs._forward_once()  # 无新数据
            self.assertEqual(sent, 0)
            fs.close()
        finally:
            center.stop()

    def test_forward_failure_keeps_cursor(self):
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
            )
            rec_path = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write('{"event_id":"a:0001","q":1}\n')
            center.fail_next = True
            fs._forward_once()  # 失败, 游标不动
            self.assertEqual(center.received, [])
            sent = fs._forward_once()  # 重试成功
            self.assertEqual(sent, 1)
            fs.close()
        finally:
            center.stop()

    def test_idempotent_after_restart(self):
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
            )
            rec_path = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write('{"event_id":"a:0001","q":1}\n')
            fs._forward_once()
            fs.close()  # 模拟重启
            fs2 = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
            )
            sent = fs2._forward_once()  # 已转发过: 幂等跳过
            self.assertEqual(sent, 0)
            fs2.close()
        finally:
            center.stop()


# ---------- 回归测试: 2026-08-19 缺陷修复(BUG-001~010) ----------


class TestSqliteStorageRegression(unittest.TestCase):
    """BUG-001: sqlite 表结构与 write() 同步瘦身后 16 字段集"""

    def _write_rec(self, storage, **overrides):
        rec = SearchLogEvent(
            query="刘月的论文用了什么模型", request_id="req_sqlite1",
            user_id="u_1", session_id="s_1", node_id="node-sqlite",
            query_cleaned="刘月的论文用了什么模型", query_primary="刘月的论文用了什么模型",
            query_subs=["顺便讲ESN"], query_rewritten="刘月的论文 用了什么模型",
        )
        for k, v in overrides.items():
            setattr(rec, k, v)
        # 直接构造 record 走存储层(不依赖 recorder 后台线程)
        from telemetry.search_log.recorder import SearchRecorder
        import time as _t
        recr = SearchRecorder(storage=storage, config=SearchLogConfig(queue_size=10, batch_size=100, flush_interval=100))
        recr.record(rec)
        recr.shutdown()
        return recr

    def test_sqlite_write_all_fields(self):
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "test.sqlite3")
        storage = SqliteStorage(db)
        recr = self._write_rec(storage)
        snap = recr.metrics.snapshot()
        self.assertEqual(snap["written"], 1, "sqlite 写入应成功")
        self.assertEqual(snap["failed"], 0)
        import sqlite3
        conn = sqlite3.connect(db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(search_log)")}
        # 瘦身字段齐全, 已删字段不在
        for c in ("query_cleaned", "query_primary", "query_subs", "query_rewritten", "session_id", "node_id"):
            self.assertIn(c, cols)
        for c in ("result_count", "latency_ms"):
            self.assertNotIn(c, cols)
        row = conn.execute(
            "SELECT event_id, query_raw, query_cleaned, query_primary, query_subs, query_rewritten, session_id, node_id FROM search_log"
        ).fetchone()
        self.assertEqual(row[1], "刘月的论文用了什么模型")
        self.assertEqual(json.loads(row[4]), ["顺便讲ESN"])
        self.assertEqual(row[6], "s_1")
        self.assertEqual(row[7], "node-sqlite")
        conn.close()

    def test_sqlite_legacy_schema_auto_migrated(self):
        """旧表(含 result_count/latency_ms)自动迁移, 公共列数据保留"""
        import sqlite3

        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "legacy.sqlite3")
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE search_log (
              event_id TEXT PRIMARY KEY, ts TEXT NOT NULL, user_id TEXT, dept_id TEXT,
              query_raw TEXT NOT NULL, query_norm TEXT NOT NULL,
              result_count INTEGER NOT NULL DEFAULT 0, latency_ms INTEGER,
              channel TEXT, request_id TEXT NOT NULL,
              sampled INTEGER NOT NULL DEFAULT 0, ip TEXT);
            INSERT INTO search_log VALUES
              ('old:0001','2026-08-01T00:00:00+0800','u','d','q','qn',3,12,'agent','old',1,'1.2.3.*');
            """
        )
        conn.commit()
        conn.close()
        SqliteStorage(db)  # 构造即迁移
        conn = sqlite3.connect(db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(search_log)")}
        self.assertNotIn("result_count", cols)
        self.assertIn("query_subs", cols)
        row = conn.execute("SELECT event_id, query_raw, query_subs FROM search_log").fetchone()
        self.assertEqual(row[0], "old:0001", "旧数据公共列应保留")
        self.assertEqual(row[1], "q")
        conn.close()


class TestForwardingCrossDayRegression(unittest.TestCase):
    """BUG-003: 跨天游标错位 → 旧文件剩余 + 新文件数据都能续转"""

    def test_cross_day_cursor_resumes_old_file_then_new(self):
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            cursor_file = os.path.join(tmp, "sub", ".forward_cursor.json")
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
                cursor_file=cursor_file,
            )
            # 昨天文件 2 行, 游标停在第 1 行结尾(昨天部分已转发)
            old_path = os.path.join(tmp, "search_log_2026-08-18.jsonl")
            line1 = '{"event_id":"old:0001","q":1}\n'
            with open(old_path, "w", encoding="utf-8") as f:
                f.write(line1 + '{"event_id":"old:0002","q":2}\n')
            fs._save_cursor(old_path, len(line1.encode("utf-8")))
            # 今天文件 1 行
            today = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(today, "w", encoding="utf-8") as f:
                f.write('{"event_id":"new:0003","q":3}\n')
            sent = fs._forward_once()
            self.assertEqual(sent, 2, "昨天剩余 1 条 + 今天 1 条都应转发")
            ids = [e["event_id"] for b in center.received for e in b]
            self.assertIn("old:0002", ids)
            self.assertIn("new:0003", ids)
            self.assertNotIn("old:0001", ids, "已转发过的不应重发")
            fs.close()
        finally:
            center.stop()

    def test_cursor_at_old_eof_then_new_file(self):
        """游标在昨天 EOF, 今天有数据 → 只转今天的"""
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            cursor_file = os.path.join(tmp, ".forward_cursor.json")
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
                cursor_file=cursor_file,
            )
            old_path = os.path.join(tmp, "search_log_2026-08-17.jsonl")
            with open(old_path, "w", encoding="utf-8") as f:
                f.write('{"event_id":"old:0001","q":1}\n')
            fs._save_cursor(old_path, os.path.getsize(old_path))  # 昨天已转完
            today = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(today, "w", encoding="utf-8") as f:
                f.write('{"event_id":"new:0002","q":2}\n')
            sent = fs._forward_once()
            self.assertEqual(sent, 1)
            ids = [e["event_id"] for b in center.received for e in b]
            self.assertEqual(ids, ["new:0002"])
            fs.close()
        finally:
            center.stop()

    def test_payload_node_id_from_config(self):
        """BUG-005: 转发 payload node_id 用配置值而非目录名"""
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            cursor_file = os.path.join(tmp, ".forward_cursor.json")
            fs = ForwardingStorage(
                tmp, forward_url=f"http://127.0.0.1:{center.port}",
                interval_s=3600, prefix="search_log", auto_start=False,
                cursor_file=cursor_file, node_id="node-9",
            )
            today = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
            with open(today, "w", encoding="utf-8") as f:
                f.write('{"event_id":"a:0001","q":1}\n')
            fs._forward_once()
            self.assertEqual(center.node_ids, ["node-9"])
            fs.close()
        finally:
            center.stop()

    def test_forwarded_pruned_by_day(self):
        """BUG-010: 幂等集按日期裁剪, 不无限增长"""
        tmp = tempfile.mkdtemp()
        fs = ForwardingStorage(
            tmp, forward_url="", interval_s=3600, prefix="search_log",
            auto_start=False, retention_days=90,
        )
        # _day_of 提取日期正确(prefix 含下划线也能切对)
        today_path = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
        self.assertEqual(fs._day_of(today_path), time.strftime("%Y-%m-%d"))
        fs._forwarded = {"a:1": "2026-01-01", "a:2": time.strftime("%Y-%m-%d")}
        fs._prune_forwarded()
        self.assertNotIn("a:1", fs._forwarded, "过期日期应被裁剪")
        self.assertIn("a:2", fs._forwarded)
        fs.close()


class TestConfigReachabilityRegression(unittest.TestCase):
    """BUG-004: 转发功能从配置可达(from_config / TelemetryBus 自动装配)"""

    def test_from_config_builds_forwarding_storage(self):
        tmp = tempfile.mkdtemp()
        cfg = SearchLogConfig(
            json_dir=tmp, forward_url="http://127.0.0.1:9",
            forward_cursor_file=os.path.join(tmp, ".cur.json"), node_id="node-x",
        )
        rec = SearchRecorder.from_config(cfg)
        self.assertIsInstance(rec.storage, ForwardingStorage)
        self.assertEqual(rec.storage.node_id, "node-x")
        rec.shutdown()

    def test_from_config_sqlite_writes_end_to_end(self):
        """BUG-001: from_config(storage='sqlite') 全链路写入"""
        import sqlite3

        tmp = tempfile.mkdtemp()
        cfg = SearchLogConfig(
            storage="sqlite", sqlite_path=os.path.join(tmp, "s.sqlite3"),
            queue_size=10, batch_size=2, flush_interval=0.05, hash_salt="salt",
        )
        rec = SearchRecorder.from_config(cfg)
        for i in range(3):
            rec.record(SearchLogEvent(query=f"q{i}", request_id=f"r{i}"))
        rec.shutdown()
        snap = rec.metrics.snapshot()
        self.assertEqual(snap["written"], 3)
        self.assertEqual(snap["failed"], 0)
        conn = sqlite3.connect(cfg.sqlite_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0], 3)
        conn.close()

    def test_telemetry_bus_auto_builds_forwarding(self):
        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            cfg = SearchLogConfig(
                json_dir=tmp, hash_salt="salt", queue_size=10, batch_size=2,
                flush_interval=0.05, forward_url=f"http://127.0.0.1:{center.port}",
                forward_cursor_file=os.path.join(tmp, ".cur.json"), node_id="node-y",
            )
            bus = TelemetryBus(node_id="node-y", config=cfg)
            self.assertIsInstance(bus._search_recorder.storage, ForwardingStorage)
            bus.emit(_make_ctx(request_id="req_fwd1"))
            bus.shutdown()
            bus._search_recorder.storage._forward_once()
            ids = [e["event_id"] for b in center.received for e in b]
            self.assertEqual(len(ids), 1)
            self.assertEqual(center.node_ids[-1], "node-y")
            bus._search_recorder.storage.close()
        finally:
            center.stop()


class TestTelemetryBusRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = SearchLogConfig(
            json_dir=self.tmp, hash_salt="salt",
            queue_size=100, batch_size=10, flush_interval=0.1,
        )

    def _read_lines(self, prefix: str) -> list[dict]:
        day = time.strftime("%Y-%m-%d")
        path = os.path.join(self.tmp, f"{prefix}_{day}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    def test_emitted_dedup_capacity_bounded(self):
        """BUG-002: 幂等集有界, 超出淘汰最旧"""
        cfg = SearchLogConfig(
            json_dir=self.tmp, hash_salt="salt", dedup_capacity=5,
            queue_size=100, batch_size=100, flush_interval=100,
        )
        bus = TelemetryBus(node_id="node-a", config=cfg)
        for i in range(10):
            bus.emit(_make_ctx(request_id=f"req_{i}"))
        self.assertLessEqual(len(bus._emitted), 5)
        bus.shutdown()

    def test_sampled_and_ts_semantics(self):
        """BUG-008/009: sampled 按采样决策置位; ts 为事件产生时间"""
        bus = TelemetryBus(node_id="node-a", config=self.config)
        bus.emit(_make_ctx(request_id="req_ts1"))
        bus.shutdown()
        rec = self._read_lines("search_log")[0]
        self.assertTrue(rec["sampled"])
        self.assertRegex(rec["ts"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}$")


class TestRecorderRegression(unittest.TestCase):
    def test_shutdown_flushes_remaining(self):
        """BUG-007: shutdown 强刷队列剩余, 不静默丢弃"""
        tmp = tempfile.mkdtemp()
        cfg = SearchLogConfig(
            json_dir=tmp, hash_salt="salt", queue_size=100,
            batch_size=1000, flush_interval=100,  # worker 不主动 flush
        )
        rec = SearchRecorder(storage=JsonFileStorage(tmp), config=cfg)
        for i in range(50):
            rec.record(SearchLogEvent(query=f"q{i}", request_id=f"r{i}"))
        rec.shutdown(timeout_s=0.5)
        snap = rec.metrics.snapshot()
        self.assertEqual(snap["written"], 50, "shutdown 应同步强刷全部积压")
        self.assertEqual(snap["failed"], 0)
        self.assertEqual(snap["dropped"], 0)

    def test_slow_writes_counter(self):
        """BUG-006: slow_writes 指标真实计数"""
        class SlowStorage:
            def write(self, records, timeout_s=2.0):
                time.sleep(0.05)

        cfg = SearchLogConfig(
            json_dir=tempfile.mkdtemp(), hash_salt="salt",
            write_timeout=0.001, queue_size=10, batch_size=5, flush_interval=0.05,
        )
        rec = SearchRecorder(storage=SlowStorage(), config=cfg)
        for i in range(5):
            rec.record(SearchLogEvent(query=f"q{i}", request_id=f"r{i}"))
        rec.shutdown()
        snap = rec.metrics.snapshot()
        self.assertEqual(snap["written"], 5)
        self.assertGreaterEqual(snap["slow_writes"], 1, "慢写应计数")


# ---------- 回归测试: 2026-08-19 二轮补测(BUG-011/012) ----------


class TestNormalizeNonStringRegression(unittest.TestCase):
    """BUG-011: normalize_query 非字符串输入不拖垮整批"""

    def test_normalize_non_string_values(self):
        from telemetry.search_log.utils import normalize_query

        self.assertEqual(normalize_query(None), "")
        self.assertEqual(normalize_query(""), "")
        self.assertEqual(normalize_query(123), "123")
        self.assertEqual(normalize_query(3.14), "3.14")
        self.assertEqual(normalize_query(["a"]), "['a']")

    def test_subs_with_non_strings_full_chain(self):
        tmp = tempfile.mkdtemp()
        cfg = SearchLogConfig(
            json_dir=tmp, hash_salt="salt", queue_size=10,
            batch_size=100, flush_interval=100,
        )
        rec = SearchRecorder(storage=JsonFileStorage(tmp), config=cfg)
        rec.record(
            SearchLogEvent(
                query="正常问题", request_id="req_ns1",
                query_subs=[None, 123, "正常"],
            )
        )
        rec.shutdown()
        snap = rec.metrics.snapshot()
        self.assertEqual(snap["written"], 1, "非字符串 subs 不应导致整批丢弃")
        self.assertEqual(snap["failed"], 0)
        day = time.strftime("%Y-%m-%d")
        with open(os.path.join(tmp, f"search_log_{day}.jsonl"), encoding="utf-8") as f:
            rec_json = json.loads(f.readline())
        self.assertEqual(rec_json["query_subs"], ["", "123", "正常"])


class TestForwardHalfLineRegression(unittest.TestCase):
    """BUG-012: 转发读到半行不推进游标, 补全后不丢"""

    def _make_fs(self, tmp, center_url=""):
        fs = ForwardingStorage(
            tmp, forward_url=center_url, interval_s=3600, prefix="search_log",
            auto_start=False, cursor_file=os.path.join(tmp, ".cur.json"),
        )
        return fs

    def test_half_line_not_forwarded_not_lost(self):
        tmp = tempfile.mkdtemp()
        posted = []
        fs = self._make_fs(tmp)
        fs._post_batch = lambda events: posted.extend(e["event_id"] for e in events)
        today = os.path.join(tmp, f"search_log_{time.strftime('%Y-%m-%d')}.jsonl")
        with open(today, "a", encoding="utf-8") as f:
            f.write('{"event_id":"a:0001","q":1}\n')
            f.write('{"event_id":"a:0002"')  # 半行
        sent = fs._forward_once()
        self.assertEqual(sent, 1)
        self.assertEqual(posted, ["a:0001"])
        # 游标停在最后一个完整行之后, 不吞半行(按实际行字节数断言, 兼容 \n/\r\n)
        with open(today, "rb") as f:
            first_line_len = len(f.readline())
        cursor = fs._load_cursor()
        self.assertEqual(cursor["offset"], first_line_len)
        # 半行补全 + 新行
        with open(today, "a", encoding="utf-8") as f:
            f.write(',"q":2}\n')
            f.write('{"event_id":"a:0003","q":3}\n')
        sent = fs._forward_once()
        self.assertEqual(sent, 2, "补全行与后续行都应转发, 不丢失")
        self.assertEqual(posted, ["a:0001", "a:0002", "a:0003"])
        fs.close()

    def test_concurrent_write_and_forward_no_loss_no_dup(self):
        """并发写 + 转发: 锁互斥 + 完整行推进, 不重不漏"""
        from telemetry.search_log.models import SearchLogRecord

        tmp = tempfile.mkdtemp()
        center = _MockCenter()
        center.start()
        try:
            fs = self._make_fs(tmp, f"http://127.0.0.1:{center.port}")
            total = 500
            batch = 10
            errors = []

            def writer():
                try:
                    for start in range(0, total, batch):
                        records = [
                            SearchLogRecord(
                                event_id=f"c:{i:04d}", ts="t", user_id=None,
                                dept_id=None, query_raw=f"q{i}", query_norm=f"q{i}",
                            )
                            for i in range(start, min(start + batch, total))
                        ]
                        fs.write(records, timeout_s=2.0)  # 与转发共享锁
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            t = threading.Thread(target=writer)
            t.start()
            while t.is_alive():
                fs._forward_once()
                time.sleep(0.002)
            # 写完成后再转发直到无新数据
            for _ in range(20):
                if fs._forward_once() == 0:
                    break
                time.sleep(0.01)
            fs.close()
            self.assertEqual(errors, [])
            ids = [e["event_id"] for b in center.received for e in b]
            self.assertEqual(len(ids), total, f"应转发 {total} 条, 实际 {len(ids)}")
            self.assertEqual(len(set(ids)), total, "不应有重复转发")
        finally:
            center.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
