# agent_limits.py - Agent 服务化防护层
# 提供: token 预算(会话/每日) / 按用户限流 / API 故障熔断与冷却 / 跨进程持久缓存
# 全部使用标准库, 持久化 JSON 原子写(跨进程安全)
from __future__ import annotations

import json
import logging
import os
import threading
import time
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============ token 估算 ============
def estimate_tokens(text: str) -> int:
    """粗略估算 token 数: CJK 每字约 1 token, 其他约 4 字符 1 token"""
    if not text:
        return 0
    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3000" <= ch <= "\u303f"
        or "\uff00" <= ch <= "\uffef"
    )
    other = max(0, len(text) - cjk)
    return cjk + other // 4


def estimate_messages_tokens(messages: list[dict[str, Any]], max_tokens: int) -> int:
    """估算一次调用消耗 = 输入(消息体+约10%开销) + 输出上限"""
    body = 0
    for message in messages:
        body += estimate_tokens(str(message.get("role", "")))
        content = message.get("content")
        if isinstance(content, str):
            body += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    body += estimate_tokens(str(item.get("text") or item.get("content") or ""))
        if "tools" in message:
            body += estimate_tokens(json.dumps(message["tools"], ensure_ascii=False)[:2000])
    return int(body * 1.1) + int(max_tokens)


# ============ 跨进程持久缓存 ============
class PersistentCache:
    """JSON 文件持久化缓存(TTL + 容量上限 + 原子写), 支持多进程读写"""

    def __init__(
        self,
        cache_file: str,
        *,
        max_entries: int = 64,
        ttl_seconds: int = 86400,
    ) -> None:
        self.cache_file = cache_file
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at = entry.get("expires_at", 0)
            if expires_at and time.time() > expires_at:
                self._data.pop(key, None)
                return None
            return entry.get("value")

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = {
                "value": value,
                "expires_at": time.time() + self.ttl_seconds,
            }
            while len(self._data) > self.max_entries:
                self._data.pop(next(iter(self._data)), None)
            self._save_locked()

    def remove(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._save_locked()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _load(self) -> None:
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                self._data = raw
        except FileNotFoundError:
            self._data = {}
        except Exception as exc:  # 损坏文件容错
            logger.warning("持久缓存加载失败(将重建): %s", exc)
            self._data = {}

    def _save_locked(self) -> None:
        try:
            directory = os.path.dirname(self.cache_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = f"{self.cache_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, ensure_ascii=False)
            os.replace(tmp_path, self.cache_file)  # 原子替换, 防并发半写
        except Exception as exc:
            logger.warning("持久缓存写入失败: %s", exc)


# ============ token 预算(会话 + 每日) ============
class TokenBudget:
    """token 预算: session(进程内, 按用户) + daily(跨进程, 持久化文件)"""

    def __init__(
        self,
        usage_file: str,
        *,
        session_limit: int = 100_000,
        daily_limit: int = 500_000,
    ) -> None:
        self.usage_file = usage_file
        self.session_limit = session_limit
        self.daily_limit = daily_limit
        self._lock = threading.Lock()
        self._session: dict[str, int] = {}
        self._daily: dict[str, dict[str, Any]] = {}
        self._usage_load_failed = False
        self._load_daily()

    def check_and_consume(
        self,
        tokens: int,
        *,
        user_id: str = "default",
    ) -> tuple[bool, str | None]:
        """检查并记账; 返回 (是否允许, 拒绝原因)"""
        if tokens <= 0:
            return True, None
        with self._lock:
            if self._usage_load_failed:
                return False, "历史 token 用量账本无法验证，已暂停模型调用；请先检查账本，勿清空重建"
            today = time.strftime("%Y-%m-%d")
            if self._daily.get("date") != today:
                self._daily = {"date": today, "users": {}}
            user_daily = self._daily["users"].setdefault(
                user_id, {"used": 0, "calls": 0}
            )
            session_used = self._session.get(user_id, 0)

            if session_used + tokens > self.session_limit:
                remaining = max(0, self.session_limit - session_used)
                return False, f"会话 token 预算已达上限(剩余约 {remaining} tokens), 请稍后重试"
            if user_daily["used"] + tokens > self.daily_limit:
                remaining = max(0, self.daily_limit - user_daily["used"])
                return False, f"今日 token 预算已达上限(剩余约 {remaining} tokens), 请明日再试"

            self._session[user_id] = session_used + tokens
            user_daily["used"] += tokens
            user_daily["calls"] += 1
            self._save_daily_locked()
            return True, None

    def snapshot(self, user_id: str = "default") -> dict[str, int]:
        with self._lock:
            user_daily = self._daily.get("users", {}).get(user_id, {})
            return {
                "session_used": self._session.get(user_id, 0),
                "session_limit": self.session_limit,
                "daily_used": int(user_daily.get("used", 0)),
                "daily_limit": self.daily_limit,
                "daily_calls": int(user_daily.get("calls", 0)),
            }

    def _load_daily(self) -> None:
        try:
            with open(self.usage_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict) or not isinstance(raw.get("date"), str) or not isinstance(raw.get("users"), dict):
                raise ValueError("invalid usage ledger structure")
            time.strptime(raw["date"], "%Y-%m-%d")
            for user, usage in raw["users"].items():
                if not isinstance(user, str) or not isinstance(usage, dict) or any(
                    type(usage.get(field)) is not int or usage[field] < 0 for field in ("used", "calls")
                ):
                    raise ValueError("invalid usage ledger counters")
            self._daily = raw
        except FileNotFoundError:
            self._daily = {"date": time.strftime("%Y-%m-%d"), "users": {}}
        except Exception:
            self._usage_load_failed = True
            logger.warning("token 用量账本无法验证，已暂停模型调用；原文件保持不变")

    def _save_daily_locked(self) -> None:
        try:
            directory = os.path.dirname(self.usage_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = f"{self.usage_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self._daily, handle, ensure_ascii=False)
            os.replace(tmp_path, self.usage_file)
        except Exception as exc:
            logger.warning("token 用量写入失败: %s", exc)


# ============ 按用户限流(固定窗口) ============
class RateLimiter:
    """按 user_id 的固定窗口限流, 持久化到文件(跨进程生效)"""

    def __init__(
        self,
        rate_file: str,
        *,
        max_calls: int = 20,
        window_seconds: int = 60,
    ) -> None:
        self.rate_file = rate_file
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def allow(self, user_id: str = "default") -> tuple[bool, float]:
        """返回 (是否放行, 需要等待秒数); 超限时等待秒数 > 0"""
        now = time.time()
        with self._lock:
            entry = self._data.setdefault(
                user_id, {"window_start": now, "count": 0}
            )
            if now - entry["window_start"] >= self.window_seconds:
                entry["window_start"] = now
                entry["count"] = 0
            if entry["count"] >= self.max_calls:
                retry_after = max(0.0, entry["window_start"] + self.window_seconds - now)
                return False, retry_after
            entry["count"] += 1
            self._save_locked()
            return True, 0.0

    def _load(self) -> None:
        try:
            with open(self.rate_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                self._data = raw
        except FileNotFoundError:
            self._data = {}
        except Exception as exc:
            logger.warning("限流状态加载失败(将重建): %s", exc)
            self._data = {}

    def _save_locked(self) -> None:
        try:
            directory = os.path.dirname(self.rate_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = f"{self.rate_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, ensure_ascii=False)
            os.replace(tmp_path, self.rate_file)
        except Exception as exc:
            logger.warning("限流状态写入失败: %s", exc)


# ============ API 熔断与冷却 ============
class CircuitOpenError(RuntimeError):
    """熔断器打开期间抛出的错误"""


class BudgetExceededError(RuntimeError):
    """token 预算超限"""


class CircuitBreaker:
    """三态熔断器: closed -> open(连续失败) -> half-open(冷却后试一次) -> closed"""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        name: str = "deepseek-api",
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1, cooldown_seconds)
        self.name = name
        self._lock = threading.Lock()
        self._state = "closed"  # closed | open | half-open
        self._failure_count = 0
        self._opened_at = 0.0

    def state(self) -> str:
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        with self._lock:
            if self._state != "open":
                return False
            if time.time() - self._opened_at >= self.cooldown_seconds:
                self._state = "half-open"
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == "half-open":
                self._state = "open"
                self._opened_at = time.time()
                return
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()
                logger.warning(
                    "[熔断] %s 连续失败 %d 次, 熔断打开, 冷却 %ds",
                    self.name, self._failure_count, self.cooldown_seconds,
                )

    def call(self, fn, *args, **kwargs):
        """熔断保护调用; open 状态直接拒绝"""
        if self.is_open():
            remaining = self.cooldown_seconds - (time.time() - self._opened_at)
            raise CircuitOpenError(
                f"API 故障熔断中(冷却剩余约 {max(1, int(remaining))}s), 请稍后再试"
            )
        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self.record_failure()
            raise


# Production cross-process implementations.  The public names are rebound at
# module end so existing callers retain their API while using SQLite WAL.
class SQLitePersistentCache:
    def __init__(self, cache_file: str, *, max_entries: int = 64, ttl_seconds: int = 86400):
        self.cache_file = cache_file
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        from agent.privacy import PersistenceCipher
        self._cipher = PersistenceCipher(str(cache_file) + ".key")
        self._prepare()

    def _connect(self):
        connection = sqlite3.connect(self.cache_file, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _prepare(self):
        path = Path(self.cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for attempt in range(8):
                try:
                    with self._connect() as connection:
                        connection.execute("PRAGMA journal_mode=WAL")
                        connection.execute("CREATE TABLE IF NOT EXISTS cache_entries (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires_at REAL NOT NULL, created_at REAL NOT NULL)")
                    break
                except sqlite3.OperationalError:
                    if attempt == 7:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except sqlite3.OperationalError:
            raise
        except sqlite3.DatabaseError:
            quarantine = path.with_name(path.name + f".corrupt-{int(time.time())}")
            path.replace(quarantine)
            with self._connect() as connection:
                connection.execute("CREATE TABLE cache_entries (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires_at REAL NOT NULL, created_at REAL NOT NULL)")

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (now,))
            row = connection.execute("SELECT payload FROM cache_entries WHERE cache_key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return self._cipher.decrypt_json(row[0], aad=f"answer-cache-v1:{key}")
        except Exception:
            self.remove(key)
            return None

    def put(self, key: str, value: Any) -> None:
        now = time.time()
        payload = self._cipher.encrypt_json(value, aad=f"answer-cache-v1:{key}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO cache_entries VALUES(?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, expires_at=excluded.expires_at, created_at=excluded.created_at", (key, payload, now + self.ttl_seconds, now))
            connection.execute("DELETE FROM cache_entries WHERE cache_key IN (SELECT cache_key FROM cache_entries ORDER BY created_at DESC LIMIT -1 OFFSET ?)", (self.max_entries,))
            connection.commit()

    def remove(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM cache_entries WHERE cache_key=?", (key,))

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM cache_entries")

    def __len__(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at > ?", (time.time(),)).fetchone()[0])


class SQLiteRateLimiter:
    def __init__(self, rate_file: str, *, max_calls: int = 20, window_seconds: int = 60):
        self.rate_file = rate_file
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        path = Path(rate_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(8):
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("CREATE TABLE IF NOT EXISTS rate_windows (user_id TEXT PRIMARY KEY, window_start REAL NOT NULL, call_count INTEGER NOT NULL)")
                break
            except sqlite3.OperationalError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _connect(self):
        connection = sqlite3.connect(self.rate_file, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def allow(self, user_id: str = "default") -> tuple[bool, float]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT window_start,call_count FROM rate_windows WHERE user_id=?", (user_id,)).fetchone()
            start, count = (float(row[0]), int(row[1])) if row else (now, 0)
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count >= self.max_calls:
                connection.commit()
                return False, max(0.0, start + self.window_seconds - now)
            connection.execute("INSERT INTO rate_windows VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET window_start=excluded.window_start, call_count=excluded.call_count", (user_id, start, count + 1))
            connection.commit()
            return True, 0.0


PersistentCache = SQLitePersistentCache
RateLimiter = SQLiteRateLimiter
