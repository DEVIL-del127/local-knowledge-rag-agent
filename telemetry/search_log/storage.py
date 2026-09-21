"""检索词记录功能 - 存储层（对应方案文档 §7）

LogStorage      : 存储接口
JsonFileStorage : JSON lines 按天轮转(默认, 零依赖, 当天即可分析)
SqliteStorage   : 零依赖 DB 批量写(演示 DB-API 批量模式; 生产可换 PostgreSQL/MySQL 的 executemany)
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Protocol

from telemetry.search_log.models import SearchLogRecord

logger = logging.getLogger(__name__)


class LogStorage(Protocol):
    def write(self, records: list[SearchLogRecord], timeout_s: float) -> None:
        """批量写; 失败抛异常(由 recorder 捕获计数, fail-silent)"""
        ...


class JsonFileStorage:
    """JSON lines 按天轮转: logs/search_log/search_log_2026-08-17.jsonl

    prefix 参数支持多流分文件(search_log / trace)。
    保留策略: 跨天时自动清理 retention_days 天前的文件。
    """

    def __init__(self, directory: str, prefix: str = "search_log", retention_days: int = 90) -> None:
        self.directory = directory
        self.prefix = prefix
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._current_day: str | None = None
        self._handle = None
        os.makedirs(directory, exist_ok=True)
        self.cleanup(retention_days)

    def write(self, records: list[Any], timeout_s: float = 2.0) -> None:
        if not records:
            return
        start = time.monotonic()
        with self._lock:
            handle = self._ensure_handle()
            for record in records:
                handle.write(record.to_json_line() + "\n")
            handle.flush()
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                logger.warning("搜索日志写入慢: %.2fs (> %.1fs)", elapsed, timeout_s)

    def _ensure_handle(self):
        day = time.strftime("%Y-%m-%d")
        if self._handle is None or day != self._current_day:
            if self._handle is not None:
                self._handle.close()
            path = os.path.join(self.directory, f"{self.prefix}_{day}.jsonl")
            self._handle = open(path, "a", encoding="utf-8")
            self._current_day = day
            self.cleanup(self.retention_days)
        return self._handle

    def cleanup(self, retention_days: int) -> None:
        """删除 retention_days 天前的文件(0=不清理)"""
        if not retention_days or retention_days <= 0:
            return
        cutoff = time.time() - retention_days * 86400
        try:
            for name in os.listdir(self.directory):
                if not name.startswith(self.prefix + "_") or not name.endswith(".jsonl"):
                    continue
                path = os.path.join(self.directory, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        logger.info("清理过期日志: %s", path)
                except OSError:
                    pass
        except OSError as exc:
            logger.warning("日志清理失败: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class ForwardingStorage(JsonFileStorage):
    """本地照常写 + 后台线程增量转发中心(断点续传, 不重不漏)

    转发协议: POST {forward_url}/batch  body={"node_id":..., "events": [...]}
    header: Authorization: Bearer {token}
    游标记录在 forward_cursor_file, 成功转发后才推进; 失败下次重试。
    """

    def __init__(
        self,
        directory: str,
        forward_url: str = "",
        token: str = "",
        interval_s: float = 3600.0,
        cursor_file: str = "",
        prefix: str = "search_log",
        retention_days: int = 90,
        node_id: str = "node-a",
        auto_start: bool = True,
    ) -> None:
        super().__init__(directory, prefix=prefix, retention_days=retention_days)
        self.forward_url = forward_url.rstrip("/")
        self.token = token
        self.interval_s = interval_s
        self.node_id = node_id
        self.cursor_file = cursor_file or os.path.join(directory, ".forward_cursor.json")
        self._stop = threading.Event()
        self._forwarded: dict[str, str] = {}  # event_id -> 所属日志文件日期(YYYY-MM-DD), 按日期裁剪防无限增长
        self._load_forwarded()
        self._thread: threading.Thread | None = None
        if auto_start and self.forward_url:
            self._thread = threading.Thread(
                target=self._forward_loop, name="search-log-forwarder", daemon=True
            )
            self._thread.start()

    # ---------- 转发循环 ----------
    def _forward_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._forward_once()
            except Exception as exc:
                logger.warning("日志转发失败(下次重试): %s", exc)

    def _forward_once(self) -> int:
        """从游标指向的文件/偏移开始, 按日期升序转发 retention 内全部未转发数据

        跨天安全: 游标记录 file+offset, 优先续读游标指向的文件(而非硬编码当天文件),
        读满一个文件后自动推进到下一个文件, 直到无更多文件。
        并发安全(BUG-012): 全程持有与本地写共享的 self._lock, 杜绝读到半行;
        且游标只推进到最后一个完整行之后, EOF 处残留半行不转发不推进, 下次续读。
        失败不推进游标(下次重试), 已转发 event_id 幂等跳过(重启不重)。
        """
        import json

        with self._lock:  # 与 JsonFileStorage.write 互斥: 转发读与本地写不并发
            files = self._list_log_files()
            if not files:
                return 0
            cursor = self._load_cursor()
            cursor_file = cursor.get("file") or ""
            offset = int(cursor.get("offset", 0) or 0)

            # 起始文件: 游标指向的文件仍在 → 从它续读; 已被 retention 清理 → 从最早文件重扫
            start_index = 0
            if cursor_file and os.path.exists(cursor_file):
                abs_files = [os.path.abspath(p) for p in files]
                if os.path.abspath(cursor_file) in abs_files:
                    start_index = abs_files.index(os.path.abspath(cursor_file))

            total = 0
            for idx in range(start_index, len(files)):
                path = files[idx]
                if os.path.abspath(path) == os.path.abspath(cursor_file):
                    off = offset
                else:
                    off = 0
                day = self._day_of(path)
                try:
                    # 二进制读: 只转发以 \n 结尾的完整行; 半行留待下次
                    with open(path, "rb") as handle:
                        handle.seek(off)
                        data = handle.read()
                except OSError as exc:
                    logger.warning("日志转发读文件失败(下次重试): %s", exc)
                    break
                if not data:
                    continue  # 无新数据, 游标不动
                if data.endswith(b"\n"):
                    complete = data
                else:
                    last_nl = data.rfind(b"\n")
                    if last_nl < 0:
                        continue  # 只有半行: 游标不动, 下次再读
                    complete = data[: last_nl + 1]
                end_offset = off + len(complete)
                lines = [
                    ln for ln in complete.decode("utf-8", "replace").splitlines() if ln.strip()
                ]
                if not lines:
                    # 无有效行: 推进到最后一个完整行之后, 继续下一个文件
                    self._save_cursor(path, end_offset)
                    continue
                events = []
                for ln in lines:
                    try:
                        rec = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event_id") in self._forwarded:
                        continue
                    events.append(rec)
                if not events:
                    # 全部已转发过: 推进到最后一个完整行之后, 避免重复读
                    self._save_cursor(path, end_offset)
                    continue
                try:
                    self._post_batch(events)
                except Exception as exc:  # 转发失败: 游标不动, 下次重试
                    logger.warning("日志转发失败(下次重试): %s", exc)
                    break
                for rec in events:
                    self._forwarded[rec.get("event_id")] = day
                self._save_cursor(path, end_offset)
                total += len(events)
            self._persist_forwarded()
            self._prune_forwarded()
            return total

    def _list_log_files(self) -> list[str]:
        """retention 目录内全部日志文件, 按日期升序(YYYY-MM-DD 字典序即日期序)"""
        try:
            names = [
                n for n in os.listdir(self.directory)
                if n.startswith(self.prefix + "_") and n.endswith(".jsonl")
            ]
        except OSError:
            return []
        names.sort()
        return [os.path.join(self.directory, n) for n in names]

    def _day_of(self, path: str) -> str:
        """从文件名提取日期: search_log_2026-08-18.jsonl → 2026-08-18"""
        base = os.path.basename(path)
        stem = base[: -len(".jsonl")]
        prefix = self.prefix
        if stem.startswith(prefix + "_"):
            return stem[len(prefix) + 1:]
        return ""

    def _post_batch(self, events: list[dict]) -> None:
        import json
        import urllib.request

        payload = json.dumps(
            {"node_id": self.node_id, "events": events},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.forward_url}/batch",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"转发被拒: HTTP {resp.status}")

    # ---------- 游标/幂等 ----------
    def _load_cursor(self) -> dict:
        import json

        try:
            with open(self.cursor_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cursor(self, path: str, offset: int) -> None:
        import json
        import os as _os

        _os.makedirs(_os.path.dirname(self.cursor_file), exist_ok=True)
        with open(self.cursor_file, "w", encoding="utf-8") as f:
            json.dump({"file": path, "offset": offset}, f, ensure_ascii=False)

    def _load_forwarded(self) -> None:
        path = self.cursor_file + ".ids"
        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    # 新格式 "event_id YYYY-MM-DD"; 兼容旧格式(单列, 日期未知)
                    parts = ln.split(" ", 1)
                    self._forwarded[parts[0]] = parts[1] if len(parts) > 1 else ""
        except OSError:
            pass

    def _persist_forwarded(self) -> None:
        path = self.cursor_file + ".ids"
        with open(path, "w", encoding="utf-8") as f:
            for eid in sorted(self._forwarded):
                f.write(f"{eid} {self._forwarded[eid]}\n")

    def _prune_forwarded(self) -> None:
        """按日期裁剪幂等集: 删除 retention 已清理文件对应的幂等记录, 防无限增长"""
        if not self.retention_days or self.retention_days <= 0:
            return
        import datetime

        cutoff = (datetime.date.today() - datetime.timedelta(days=self.retention_days)).isoformat()
        stale = [eid for eid, day in self._forwarded.items() if day and day < cutoff]
        if stale:
            for eid in stale:
                del self._forwarded[eid]
            logger.info("清理过期幂等记录 %d 条", len(stale))

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        super().close()


class SqliteStorage:
    """SQLite 批量写(零依赖演示); 建表与方案文档 §5 一致

    生产替换: 将 _connect/executemany 换成 PostgreSQL/MySQL 连接池。
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS search_log (
      event_id       TEXT PRIMARY KEY,
      ts             TEXT NOT NULL,
      user_id        TEXT,
      dept_id        TEXT,
      query_raw      TEXT NOT NULL,
      query_norm     TEXT NOT NULL,
      query_cleaned  TEXT NOT NULL DEFAULT '',
      query_primary  TEXT NOT NULL DEFAULT '',
      query_subs     TEXT NOT NULL DEFAULT '[]',
      query_rewritten TEXT NOT NULL DEFAULT '',
      channel        TEXT,
      request_id     TEXT NOT NULL,
      sampled        INTEGER NOT NULL DEFAULT 0,
      ip             TEXT,
      session_id     TEXT NOT NULL DEFAULT '',
      node_id        TEXT NOT NULL DEFAULT ''
      ,execution_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_search_log_ts    ON search_log (ts);
    CREATE INDEX IF NOT EXISTS idx_search_log_dept  ON search_log (dept_id, ts);
    CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log (query_norm);
    """

    # 瘦身后落库字段集(与 models.SearchLogRecord 对齐, 不含 result_count/latency_ms)
    COLUMNS = (
        "event_id", "ts", "user_id", "dept_id",
        "query_raw", "query_norm", "query_cleaned", "query_primary",
        "query_subs", "query_rewritten", "channel", "request_id",
        "sampled", "ip", "session_id", "node_id", "execution_json",
    )

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn:
            with conn:
                conn.executescript(self.SCHEMA)
                self._migrate_if_needed(conn)

    def _migrate_if_needed(self, conn: sqlite3.Connection) -> None:
        """旧库(含已删除的 result_count/latency_ms 或缺瘦身列)自动迁移到新结构

        策略: 旧表重命名 → 建新表 → 拷贝公共列 → 删旧表, 不丢已有数据。
        """
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(search_log)")}
        except sqlite3.Error:
            return
        if not cols:
            return
        if set(self.COLUMNS).issubset(cols) and "result_count" not in cols:
            return  # 已是新结构
        logger.warning("检测到旧版 search_log 表结构, 自动迁移(保留公共列数据)")
        conn.execute("ALTER TABLE search_log RENAME TO search_log_legacy")
        conn.executescript(self.SCHEMA)
        common = [c for c in self.COLUMNS if c in cols]
        if common:
            col_sql = ", ".join(common)
            conn.execute(
                f"INSERT INTO search_log ({col_sql}) SELECT {col_sql} FROM search_log_legacy"
            )
        conn.execute("DROP TABLE search_log_legacy")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        return conn

    def write(self, records: list[SearchLogRecord], timeout_s: float = 2.0) -> None:
        if not records:
            return
        import json

        start = time.monotonic()
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                "INSERT OR IGNORE INTO search_log "
                "(event_id, ts, user_id, dept_id, query_raw, query_norm, "
                " query_cleaned, query_primary, query_subs, query_rewritten, "
                " channel, request_id, sampled, ip, session_id, node_id, execution_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.event_id, r.ts, r.user_id, r.dept_id,
                        r.query_raw, r.query_norm, r.query_cleaned,
                        r.query_primary, json.dumps(r.query_subs, ensure_ascii=False),
                        r.query_rewritten, r.channel, r.request_id,
                        1 if r.sampled else 0, r.ip,
                        r.session_id, r.node_id,
                        json.dumps(r.execution, ensure_ascii=False, sort_keys=True),
                    )
                    for r in records
                ],
            )
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                logger.warning("搜索日志 DB 写入慢: %.2fs (> %.1fs)", elapsed, timeout_s)

def build_storage(config) -> LogStorage:
    """按配置选择存储: sqlite | forwarding(配了 forward_url) | json

    集中存放供 SearchRecorder.from_config 与 TelemetryBus 共用,
    保证配置了 forward_url 时转发真正可达(BUG-004)。
    """
    if config.storage == "sqlite":
        return SqliteStorage(config.sqlite_path)
    if config.forward_url:
        return ForwardingStorage(
            config.json_dir,
            forward_url=config.forward_url,
            token=config.forward_token,
            interval_s=config.forward_interval_s,
            cursor_file=config.forward_cursor_file,
            prefix="search_log",
            retention_days=config.retention_days,
            node_id=config.node_id,
        )
    return JsonFileStorage(
        config.json_dir, prefix="search_log", retention_days=config.retention_days
    )
