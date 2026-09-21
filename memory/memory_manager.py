# memory_manager.py - 记忆管理器（M1: 写入链路 + 会话生命周期）
# 对应《Agent 记忆管理设计 v2.0》§3 + 《任务拆分 v1.1》
# 职责: 提取管线(清洗→LLM提取→去重→冲突→落盘) / 会话生命周期(启动/结束晋级) / 会话摘要
# 说明: 召回链路(M2)未实现, ingest 为后台管线(user-invocable=false)
from __future__ import annotations

import json
import hashlib
import logging
import os
import sqlite3
import time
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from memory.memory_cleaning import anchor_conflict, clean_text, literal_dedup, verify_fidelity
from agent.agent_models import IntentPrediction
from agent.privacy import DataSanitizer, PersistenceCipher

logger = logging.getLogger(__name__)

# prompts/ 在项目根(与 memory/ 平级)
PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
)


@dataclass
class MemoryConfig:
    extract_every_n_rounds: int = 2        # 提取频率(轮)
    promote_confidence: float = 0.7        # L2 晋级置信度门槛
    summary_max_chars: int = 500           # 摘要长度上限
    max_fact_len: int = 80                 # 单条事实上限
    session_timeout_hours: int = 24        # 超时会话判定
    raw_fallback_chars: int = 200          # 提取失败原文兜底长度
    summarize_on_exit: bool = False        # 退出路径不得额外消耗外部模型额度


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write(path: str, data: Any) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(prefix=".memory-", suffix=".tmp", dir=directory or ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except Exception as exc:
        logger.warning("记忆文件加载失败(将重建): %s: %s", path, exc)
        return default


class MemoryManager:
    """记忆管理器: 写入链路 + 会话生命周期（M1 范围）"""

    def __init__(
        self,
        llm_client=None,
        *,
        state_dir: str = "agent_state",
        config: MemoryConfig | None = None,
        embedder=None,
    ) -> None:
        self.llm_client = llm_client  # DeepSeekClient(可空: 无 LLM 时降级 raw)
        self.config = config or MemoryConfig()
        self.state_dir = state_dir
        self.sessions_dir = os.path.join(state_dir, "sessions")
        self.memory_dir = os.path.join(state_dir, "memory")
        self.embedder = embedder  # bge-m3 embedder(可空: 无向量检索)
        self._collection = None  # ChromaDB agent_memory 惰性初始化
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._memory_db = os.path.join(self.state_dir, "memory", "memory.sqlite3")
        self._cipher = PersistenceCipher(self._memory_db + ".key")
        self._init_memory_db()
        self._extract_prompt = self._read_prompt("memory_write_rules.md")
        self._summary_prompt = self._read_prompt("memory_summary.md")
        # Startup never performs vector projection or implicit data maintenance.

    def _memory_connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self._memory_db), exist_ok=True)
        conn = sqlite3.connect(self._memory_db, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_memory_db(self) -> None:
        with self._memory_connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_fact(
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, revision INTEGER NOT NULL,
              payload_json TEXT NOT NULL, tombstoned INTEGER NOT NULL DEFAULT 0,
              updated_ts TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_user
              ON memory_fact(user_id, tombstoned, updated_ts);
            CREATE TABLE IF NOT EXISTS memory_session(
              session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
              revision INTEGER NOT NULL, archived INTEGER NOT NULL DEFAULT 0,
              payload_json TEXT NOT NULL, updated_ts TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_request_receipt(event_id TEXT PRIMARY KEY);
            CREATE INDEX IF NOT EXISTS idx_memory_session_user
              ON memory_session(user_id, archived);
            CREATE TABLE IF NOT EXISTS memory_outbox(
              idempotency_key TEXT PRIMARY KEY, user_id TEXT NOT NULL, fact_id TEXT NOT NULL,
              revision INTEGER NOT NULL, operation TEXT NOT NULL, payload_json TEXT NOT NULL,
              created_ts TEXT NOT NULL, completed_ts TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT
            );
            """)

    def _migrate_legacy_l2(self) -> None:
        """One-way, idempotent import of existing JSONL facts into SQLite."""
        if not os.path.isdir(self.memory_dir):
            return
        for user_id in os.listdir(self.memory_dir):
            path = os.path.join(self.memory_dir, user_id, "l2", "memory.jsonl")
            if not os.path.isfile(path):
                continue
            try:
                entries = []
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if isinstance(value, str):
                            value = self._cipher.decrypt_json(
                                value, aad=f"memory-l2-v1:{user_id}"
                            )
                        if isinstance(value, dict):
                            entries.append(value)
                with self._memory_connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for entry in entries:
                        fact_id = str(entry.get("id") or "")
                        if not fact_id:
                            continue
                        payload = dict(entry, revision=int(entry.get("revision", 1)), user_id=user_id)
                        conn.execute(
                            "INSERT OR IGNORE INTO memory_fact(id,user_id,revision,payload_json,tombstoned,updated_ts) "
                            "VALUES(?,?,?,?,0,?)",
                            (fact_id, user_id, payload["revision"], self._cipher.encrypt_json(
                                payload,
                                aad=f"memory-fact-v1:{user_id}:{fact_id}:{payload['revision']}",
                            ),
                             str(entry.get("ts") or _now_iso())),
                        )
            except Exception as exc:
                logger.warning("旧 L2 JSONL 迁移失败(%s): %s", path, exc)

    def apply_request_event(self, event_id: str, event: dict) -> None:
        self._commit_mutations(event["user_id"], event["mutations"], event_id=event_id)

    def _commit_mutations(self, user_id: str, mutations: list[tuple[str, dict]], *, event_id=None) -> None:
        """Commit fact state and vector intent atomically; vector I/O happens after commit."""
        from agent.model_execution_context import current_operation
        active = current_operation()
        if active is not None and event_id is None:
            active[0].stage_memory_mutations(active[1], user_id, mutations)
            return
        with self._memory_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if event_id is not None:
                inserted = conn.execute("INSERT OR IGNORE INTO memory_request_receipt VALUES(?)", (event_id,)).rowcount
                if not inserted:
                    return
            for operation, entry in mutations:
                fact_id = str(entry["id"])
                old = conn.execute(
                    "SELECT revision,payload_json FROM memory_fact WHERE id=? AND user_id=?", (fact_id, user_id)
                ).fetchone()
                if old and entry.get("relation") == "explicit_memory" and entry.get("operation_id"):
                    prior = self._cipher.decrypt_json(
                        old[1], aad=f"memory-fact-v1:{user_id}:{fact_id}:{old[0]}",
                    )
                    if (operation == "upsert" and prior.get("operation_id") == entry["operation_id"]
                            and prior.get("value") == entry.get("value")):
                        continue
                revision = (int(old[0]) + 1) if old else int(entry.get("revision", 1))
                payload = dict(entry, revision=revision, user_id=user_id)
                encrypted = self._cipher.encrypt_json(
                    payload, aad=f"memory-fact-v1:{user_id}:{fact_id}:{revision}"
                )
                tombstoned = 1 if operation == "delete" else 0
                conn.execute(
                    "INSERT INTO memory_fact(id,user_id,revision,payload_json,tombstoned,updated_ts) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,"
                    "payload_json=excluded.payload_json,tombstoned=excluded.tombstoned,updated_ts=excluded.updated_ts",
                    (fact_id, user_id, revision, encrypted, tombstoned, _now_iso()),
                )
                key = f"{user_id}:{fact_id}:{revision}:{operation}"
                conn.execute(
                    "INSERT OR IGNORE INTO memory_outbox(idempotency_key,user_id,fact_id,revision,operation,"
                    "payload_json,created_ts) VALUES(?,?,?,?,?,?,?)",
                    (key, user_id, fact_id, revision, operation,
                     self._cipher.encrypt_json(payload, aad=f"memory-outbox-v1:{key}"),
                     _now_iso()),
                )
        if event_id is None:
            self.replay_outbox()

    def replay_outbox(self, *, limit: int = 200) -> int:
        """Idempotently project committed memory mutations into the vector index."""
        with self._memory_connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_outbox WHERE completed_ts IS NULL ORDER BY created_ts LIMIT ?", (limit,)
            ).fetchall()
        completed = 0
        for row in rows:
            try:
                payload = self._cipher.decrypt_json(
                    row["payload_json"], aad=f"memory-outbox-v1:{row['idempotency_key']}"
                )
                if row["operation"] == "delete":
                    self._vector_delete(row["fact_id"], row["user_id"], strict=True)
                else:
                    if self.embedder is None:
                        # No vector projection is configured; relational state is authoritative.
                        pass
                    else:
                        self._vectorize([payload], row["user_id"], strict=True)
                with self._memory_connect() as conn:
                    conn.execute(
                        "UPDATE memory_outbox SET completed_ts=?,attempts=attempts+1,last_error=NULL "
                        "WHERE idempotency_key=? AND completed_ts IS NULL", (_now_iso(), row["idempotency_key"]),
                    )
                completed += 1
            except Exception as exc:
                with self._memory_connect() as conn:
                    conn.execute(
                        "UPDATE memory_outbox SET attempts=attempts+1,last_error=? WHERE idempotency_key=?",
                        (str(DataSanitizer.sanitize(
                            f"{type(exc).__name__}: {exc}"
                        ))[:1000], row["idempotency_key"]),
                    )
                logger.warning("记忆 outbox 投影失败，将在下次启动重放: %s", exc)
        return completed

    # ---------- Prompt 加载 ----------
    def _read_prompt(self, name: str) -> str:
        path = os.path.join(PROMPTS_DIR, name)
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            logger.warning("prompt 文件缺失: %s", path)
            return ""

    # ============ 会话生命周期 ============
    def on_session_start(self, *, user_id: str) -> str:
        """Create this session only; startup must not migrate or promote old data."""
        session_id = f"s_{user_id}_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
        session = {
            "schema_version": 1,
            "session_id": session_id, "user_id": user_id,
            "created_ts": _now_iso(), "updated_ts": _now_iso(),
            "ended_ts": None, "end_reason": None,
            "facts": [], "task_chains": [], "early_summary": None,
            "profile_candidates": [], "stats": {"rounds": 0, "extractions": 0},
        }
        self._save_session(session)
        return session_id

    def _recover_crashed_sessions(self, user_id: str) -> None:
        """扫描未归档会话: 24h 未更新或未正常结束 → 补晋级并归档"""
        for name in os.listdir(self.sessions_dir):
            if not name.endswith(".json"):
                continue
            session = self._load_session(os.path.splitext(name)[0])
            if not session or session.get("ended_ts"):
                continue
            if session.get("user_id") != user_id:
                continue
            updated = session.get("updated_ts", "")
            if self._is_stale(updated):
                logger.warning("检测到未正常结束的会话 %s, 补执行晋级", name)
                self._finalize_session(session, reason="crash")

    @staticmethod
    def _is_stale(iso_ts: str) -> bool:
        try:
            ts = time.mktime(time.strptime(iso_ts, "%Y-%m-%dT%H:%M:%S%z")[:9])
            return (time.time() - ts) > 24 * 3600
        except (ValueError, TypeError):
            return False

    def on_session_end(self, *, session_id: str, user_id: str, reason: str = "normal") -> None:
        """会话结束: 摘要 → L2 晋级 → 画像更新 → 归档"""
        session = self._load_session(session_id)
        if not session:
            return
        if session.get("user_id") != user_id:
            raise ValueError("memory_session_user_mismatch")
        self._finalize_session(session, reason=reason)

    def abort_session(self, *, session_id: str, user_id: str) -> None:
        """Close a partially started session without extraction or promotion."""
        session = self._load_session(session_id)
        if session is None:
            return
        if session.get("user_id") != user_id:
            raise ValueError("memory_session_user_mismatch")
        session.update(ended_ts=_now_iso(), end_reason="startup_failed", archived=True)
        self._save_session(session)

    def _finalize_session(self, session: dict, *, reason: str) -> None:
        session["ended_ts"] = _now_iso()
        session["end_reason"] = reason
        # Explicit facts already entered through the request transaction.
        # Exit is one durable archive update, never a second mutation pipeline.
        session["archived"] = True
        self._save_session(session)
        logger.info("会话 %s 已归档(reason=%s)", session["session_id"], reason)

    # ============ 每轮写入链路 ============
    def ingest_message(
        self,
        *,
        round_no: int,
        user_msg: str,
        agent_reply: str,
        intent: IntentPrediction | None,
        session_id: str,
        user_id: str,
        force: bool = False,
    ) -> None:
        """Compatibility hook: passive turns never authorize persistent memory."""
        # Passive conversation is not permission to retain facts. Explicit saves
        # are executed by admin(save), within the request's memory command path.
        return

    @staticmethod
    def _admit_turn(intent: IntentPrediction | None) -> bool:
        """Reject control, clarification and failed turns before fact extraction."""
        if intent is None:  # compatibility for direct/internal callers
            return True
        value = str(getattr(getattr(intent, "intent", None), "value", ""))
        reason = str(getattr(intent, "reason", ""))
        if value in {"clarify", "unsupported"}:
            return False
        if reason.startswith(("dialogue_", "runtime_operation_")) or reason in {
            "confirmation_without_pending", "runtime_clarification",
            "runtime_operation_replayed",
            "semantic_not_executable", "invalid_observation", "generation_mismatch",
            "no_match", "document_absent",
            "pending_action_cancelled", "clarification_expired", "resume_snapshot_unavailable",
            "general_chat_provider_unavailable", "general_chat_generation_failed",
        }:
            return False
        return True

    def _extract_and_store(self, session: dict, user_msg: str, agent_reply: str, round_no: int) -> None:
        # ① L1 代码清洗 + 保真校验
        cleaned = clean_text(user_msg)
        ok, missing = verify_fidelity(user_msg, cleaned.text)
        if not ok:
            logger.warning("清洗保真校验失败(missing=%s), 原文入 raw", missing)

        if self.llm_client is None:
            extracted = self._fallback_raw(user_msg, cleaned.text)
        else:
            extracted = self._extract_with_llm(user_msg, cleaned, session)
            if extracted is None:
                extracted = self._fallback_raw(user_msg, cleaned.text)

        # ② 去重 + 冲突
        existing = session.get("facts", [])
        new_facts = literal_dedup(extracted.get("facts", []), existing)
        new_facts = anchor_conflict(new_facts, existing)
        for fact in new_facts:
            fact.setdefault("id", f"f_{uuid.uuid4().hex[:8]}")
            fact["source_ts"] = _now_iso()
            fact["source_round"] = round_no
        session["facts"] = existing + new_facts

        # ③ 任务链
        for step in extracted.get("task_chain", []):
            step.setdefault("round", round_no)
            step["result"] = step.get("result", "ok")
            self._append_task_step(session, step)

        # ④ 画像候选
        session["profile_candidates"] = session.get("profile_candidates", []) + [
            {**c, "ts": _now_iso()} for c in extracted.get("profile_candidates", [])
        ]

    def _extract_with_llm(self, user_msg: str, cleaned, session: dict) -> dict | None:
        """P1 提取 prompt 调用; 返回 dict 或 None(失败)"""
        try:
            existing = json.dumps(
                [{"type": f.get("type"), "content": f.get("content")} for f in session.get("facts", [])[-10:]],
                ensure_ascii=False,
            )
            recent = json.dumps(
                [{"role": "user", "content": user_msg[:500]}],
                ensure_ascii=False,
            )
            text = self.llm_client.invoke_text(
                system_prompt=self._extract_prompt,
                user_prompt=(
                    f"最近对话: {recent}\n"
                    f"已有事实(用于去重): {existing}\n"
                    "请输出提取结果 JSON。"
                ),
                max_tokens=1024,
            )
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception as exc:
            logger.warning("LLM 提取失败, 降级 raw: %s", exc)
            return None

    def _fallback_raw(self, user_msg: str, cleaned_text: str) -> dict:
        """提取失败降级: 原文截断入 raw 桶"""
        return {
            "facts": [{
                "type": "raw",
                "content": cleaned_text[: self.config.raw_fallback_chars],
                "confidence": 0.3,
            }],
            "task_chain": [],
            "profile_candidates": [],
        }

    def _append_task_step(self, session: dict, step: dict) -> None:
        chains = session.setdefault("task_chains", [])
        # 找 active 链(最近一条)追加; 无则新建
        active = next((c for c in reversed(chains) if c.get("status") == "active"), None)
        if active is None:
            active = {
                "chain_id": f"c_{uuid.uuid4().hex[:8]}",
                "label": str(step.get("query", "任务"))[:30],
                "status": "active",
                "created_round": step.get("round", 0),
                "steps": [],
            }
            chains.append(active)
        active.setdefault("steps", []).append(step)

    # ============ 会话摘要 + 晋级 ============
    def _summarize_session(self, session: dict) -> str | None:
        """P2 四段式摘要; 失败返回 None(调用方兜底)"""
        if self.llm_client is None:
            return None
        try:
            data = json.dumps({
                "facts": session.get("facts", [])[:50],
                "task_chains": session.get("task_chains", [])[:10],
            }, ensure_ascii=False)
            text = self.llm_client.invoke_text(
                system_prompt=self._summary_prompt,
                user_prompt=f"会话数据: {data}",
                max_tokens=1024,
            )
            payload = json.loads(text)
            if not isinstance(payload, dict):
                return None
            parts = [
                f"任务: {payload.get('task', '[无]')}",
                f"结论: {payload.get('conclusion', '[无]')}",
                f"偏好: {payload.get('preference', '[无]')}",
                f"待办: {payload.get('todo', '[无]')}",
            ]
            return "\n".join(parts)[: self.config.summary_max_chars]
        except Exception as exc:
            logger.warning("会话摘要失败: %s", exc)
            return None

    def _promote_to_l2(self, session: dict, summary: str | None, user_id: str) -> list[dict]:
        """L2 晋级: confidence≥门槛的事实 + 摘要 → memory/{user_id}/l2/memory.jsonl + 向量"""
        user_memory_dir = os.path.join(self.memory_dir, user_id, "l2")
        os.makedirs(user_memory_dir, exist_ok=True)
        promoted: list[dict] = []

        for fact in session.get("facts", []):
            if fact.get("explicit_consent") is not True:
                continue
            if fact.get("type") == "raw":
                continue
            if float(fact.get("confidence", 0)) < self.config.promote_confidence:
                continue
            if fact.get("conflicts_with"):
                continue
            entry = {
                "id": f"fact_{uuid.uuid4().hex[:12]}",
                "type": "fact",
                "entity": _guess_entity(fact.get("content", "")),
                "relation": _type_to_relation(fact.get("type")),
                "value": fact.get("content", "")[: self.config.max_fact_len],
                "ts": fact.get("source_ts", _now_iso()),
                "session_id": session.get("session_id"),
                "source_round": fact.get("source_round"),
                "confidence": fact.get("confidence"),
                "retrieval_count": 0,
            }
            promoted.append(entry)

        if summary and session.get("summary_consent") is True:
            entry = {
                "id": f"summary_{uuid.uuid4().hex[:12]}",
                "type": "summary",
                "entity": session.get("session_id"),
                "relation": "session_summary",
                "value": summary,
                "ts": session.get("ended_ts", _now_iso()),
                "session_id": session.get("session_id"),
                "source_round": None,
                "confidence": 0.6,
                "retrieval_count": 0,
            }
            promoted.append(entry)
        if promoted:
            self._commit_mutations(user_id, [("upsert", item) for item in promoted])
        return promoted

    # ---------- 向量索引(ChromaDB agent_memory) ----------
    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        if self.embedder is None:
            return None
        try:
            import chromadb

            client = chromadb.PersistentClient(
                path=os.path.join(self.state_dir, "memory_vector")
            )
            self._collection = client.get_or_create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning("记忆向量库初始化失败(降级关键词检索): %s", exc)
            self._collection = None
        return self._collection

    def _vectorize(self, entries: list[dict], user_id: str, *, strict: bool = False) -> None:
        collection = self._ensure_collection()
        if collection is None or not entries:
            if strict and entries and self.embedder is not None:
                raise RuntimeError("memory vector collection unavailable")
            return
        try:
            texts = [f"{e.get('entity', '')} {e.get('relation', '')} {e.get('value', '')}" for e in entries]
            embeddings = self.embedder.embed(texts)
            valid = [(e, emb) for e, emb in zip(entries, embeddings) if emb is not None]
            if not valid:
                return
            collection.upsert(
                ids=[e["id"] for e, _ in valid],
                embeddings=[emb for _, emb in valid],
                # Chroma persists documents in plaintext.  Keep only the
                # opaque fact id there; semantic information is represented by
                # the embedding and hydrated from encrypted SQLite after hit.
                documents=[str(e["id"]) for e, _ in valid],
                metadatas=[{
                    "type": e.get("type", "fact"),
                    "ts": e.get("ts", ""),
                    "user_id": user_id,
                } for e, _ in valid],
            )
        except Exception as exc:
            if strict:
                raise
            logger.warning("记忆向量化失败(不影响结构化存储): %s", exc)

    def _update_profile(self, session: dict, user_id: str) -> None:
        """L3 画像: 增量合并(新值0.5起步, 重复+0.1, 冲突连续2次覆盖)"""
        candidates = session.get("profile_candidates", [])
        if not candidates:
            return
        profile_path = os.path.join(self.memory_dir, user_id, "profile.json")
        profile = self._load_profile(user_id)

        for cand in candidates:
            if cand.get("explicit_consent") is not True:
                continue
            field = str(cand.get("field", ""))
            value = str(cand.get("value", "")).strip()
            if not field or not value:
                continue
            conf = float(cand.get("confidence", 0.5))
            self._merge_profile_field(profile, field, value, conf, cand.get("ts", _now_iso()))

        _atomic_write(profile_path, self._cipher.encrypt_json(
            profile, aad=f"memory-profile-v1:{user_id}"
        ))

    def _load_profile(self, user_id: str) -> dict:
        profile_path = os.path.join(self.memory_dir, user_id, "profile.json")
        value = _load_json(profile_path, None)
        if isinstance(value, str):
            decoded = self._cipher.decrypt_json(
                value, aad=f"memory-profile-v1:{user_id}"
            )
            if isinstance(decoded, dict):
                return decoded
        if isinstance(value, dict):  # legacy plaintext migration compatibility
            return value
        return {"schema_version": 1, "stable": {}, "dynamic": {}, "history": {}}

    @staticmethod
    def _merge_profile_field(profile: dict, field: str, value: str, conf: float, ts: str) -> None:
        section = "stable" if field in {"identity", "domain", "preference_style"} else "dynamic"
        current = profile[section].get(field)
        if current is None:
            profile[section][field] = {"value": value, "confidence": conf, "updated_ts": ts, "conflicts": []}
            return
        if current.get("value") == value:
            current["confidence"] = min(1.0, float(current.get("confidence", 0.5)) + 0.1)
            current["updated_ts"] = ts
            return
        # 冲突: 记录; 连续 2 次新值才覆盖
        conflicts = current.setdefault("conflicts", [])
        same_new = [c for c in conflicts if c.get("new") == value]
        if same_new:
            same_new[0]["second_seen_ts"] = ts
            same_new[0]["resolved"] = True
            current["value"] = value
            current["confidence"] = conf
            current["updated_ts"] = ts
        else:
            conflicts.append({"old": current.get("value"), "new": value,
                              "first_seen_ts": ts, "second_seen_ts": None, "resolved": False})

    # ============ 召回链路(M2) ============
    def recall_for_current(
        self,
        *,
        query: str,
        window_messages: Sequence[str],
        session_id: str,
        user_id: str,
        budget_tokens: int = 400,
        has_anaphora_hint: bool = False,
    ) -> str | None:
        """会话内召回: 指代检测命中 → 检索 L1 → 注入文本; 未命中返回 None"""
        from memory.memory_retrieval import (
            detect_anaphora,
            extract_entities,
            format_injection,
            keyword_search,
            vector_search,
        )

        session = self._load_session(session_id)
        if not session:
            return None
        facts = session.get("facts", [])
        if not facts:
            return None

        # ① 触发检测: 规则级 + LLM 级(hint)
        if not has_anaphora_hint and not detect_anaphora(query, list(window_messages)):
            return None

        # ② 实体提取(带降级: 无实体取最近 3 条事实)
        entities = extract_entities(query)
        if len(facts) >= 500:
            collection = self._ensure_collection()
            candidates = vector_search(collection, self.embedder, query, top_k=5,
                                       where={"user_id": user_id}) if collection is not None else []
            by_id = {str(item.get("id")): item for item in facts}
            candidates = [by_id[str(item.get("id"))] for item in candidates
                          if str(item.get("id")) in by_id]
        elif entities:
            candidates = keyword_search(facts, entities, top_k=5)
        else:
            candidates = facts[-3:]
        # 实体匹配为空(如"那个模型"粘连指代) → 兜底最近事实
        if not candidates:
            candidates = facts[-3:]

        # ④ 注入
        if not candidates:
            return None
        return format_injection("会话记忆", candidates, budget_tokens=budget_tokens)

    def recall_memory(
        self,
        *,
        query: str,
        user_id: str,
        budget_tokens: int = 800,
    ) -> str:
        """跨会话召回(MEMORY_RECALL): L2 关键词/向量 → 注入文本(带来源标注)"""
        from memory.memory_retrieval import extract_entities, format_injection, vector_search

        entries = self._load_l2(user_id)
        if not entries:
            return ""

        entities = extract_entities(query)
        from memory.memory_retrieval import keyword_search

        candidates = keyword_search(entries, entities, top_k=5)
        if not candidates:
            collection = self._ensure_collection()
            if collection is not None:
                candidates = vector_search(
                    collection, self.embedder, query, top_k=5,
                    where={"user_id": user_id},
                )
                by_id = {str(item.get("id")): item for item in entries}
                candidates = [by_id[str(item.get("id"))] for item in candidates
                              if str(item.get("id")) in by_id]

        # 时间排序 + 来源标注
        candidates.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
        source_label = "、".join(
            f"{c.get('ts', '')[:10]} 会话" for c in candidates[:2]
        ) or None
        return format_injection(
            "长期记忆", candidates, budget_tokens=budget_tokens,
            source_label=source_label,
        )

    def _load_l2(self, user_id: str) -> list[dict]:
        """Read authoritative facts; empty/error must never resurrect a legacy export."""
        with self._memory_connect() as conn:
            rows = conn.execute(
                "SELECT id,revision,payload_json FROM memory_fact "
                "WHERE user_id=? AND tombstoned=0 ORDER BY updated_ts,id", (user_id,),
            ).fetchall()
        return [self._cipher.decrypt_json(
            row["payload_json"], aad=f"memory-fact-v1:{user_id}:{row['id']}:{row['revision']}",
        ) for row in rows]

    # ============ 记忆管理(M3 admin) ============
    def admin(self, *, command: str, user_id: str, args: dict | None = None) -> dict:
        """记忆管理命令: list / get / delete / fix / export / clear / stats / archive_stale"""
        args = args or {}
        if command == "save":
            value = args.get("value")
            if args.get("explicit_consent") is not True or not isinstance(value, str) or not 1 <= len(value.strip()) <= 500:
                raise ValueError("保存记忆需要明确授权和 1–500 字的内容。")
            from agent.model_execution_context import current_operation
            operation = current_operation()
            identity = operation[1].operation_id if operation else uuid.uuid4().hex
            fact_id = "fact_" + hashlib.sha256(f"{user_id}:{identity}:save".encode()).hexdigest()[:32]
            entry = {"id": fact_id, "type": "fact", "entity": "user",
                     "relation": "explicit_memory", "value": value.strip(),
                     "confidence": 1.0, "explicit_consent": True,
                     "ts": _now_iso(), "operation_id": identity}
            self._commit_mutations(user_id, [("upsert", entry)])
            return {"id": fact_id, "value": entry["value"]}
        entries = self._load_l2(user_id)

        if command == "list":
            limit = int(args.get("limit", 20))
            return {"total": len(entries), "items": entries[-limit:]}

        if command == "get":
            target_id = str(args.get("id", ""))
            for entry in entries:
                if entry.get("id") == target_id:
                    return {"item": entry}
            return {"error": f"未找到记忆 {target_id}"}

        if command == "delete":
            target_id = str(args.get("id", ""))
            kept = [e for e in entries if e.get("id") != target_id]
            if len(kept) == len(entries):
                return {"error": f"未找到记忆 {target_id}"}
            target = next(e for e in entries if e.get("id") == target_id)
            self._commit_mutations(user_id, [("delete", target)])
            return {"deleted": target_id, "remaining": len(kept)}

        if command == "fix":
            target_id = str(args.get("id", ""))
            new_value = str(args.get("value", "")).strip()
            for entry in entries:
                if entry.get("id") == target_id and new_value:
                    entry["value"] = new_value
                    self._commit_mutations(user_id, [("upsert", entry)])
                    return {"fixed": target_id, "value": new_value}
            return {"error": f"未找到记忆 {target_id}"}

        if command == "export":
            return {"exported": entries}

        if command == "clear":
            self._commit_mutations(user_id, [("delete", entry) for entry in entries])
            return {"cleared": True}

        if command == "stats":
            total = len(entries)
            by_type: dict[str, int] = {}
            conf_sum = 0.0
            for entry in entries:
                by_type[entry.get("type", "?")] = by_type.get(entry.get("type", "?"), 0) + 1
                conf_sum += float(entry.get("confidence", 0))
            return {
                "total": total,
                "by_type": by_type,
                "avg_confidence": round(conf_sum / total, 3) if total else 0,
                "retrieved_total": sum(int(e.get("retrieval_count", 0)) for e in entries),
            }

        if command == "archive_stale":
            # 30 天未检索 且 置信度 < 0.7 → 移入 archived 子目录
            stale = []
            kept = []
            import datetime as _dt

            cutoff = (time.time() - 30 * 86400)
            for entry in entries:
                ts = entry.get("ts", "")
                try:
                    ts_epoch = time.mktime(
                        _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z").timetuple()
                    )
                except (ValueError, TypeError):
                    ts_epoch = 0
                if (
                    int(entry.get("retrieval_count", 0)) == 0
                    and ts_epoch < cutoff
                    and float(entry.get("confidence", 0)) < 0.7
                ):
                    stale.append(entry)
                else:
                    kept.append(entry)
            self._commit_mutations(user_id, [("delete", entry) for entry in stale])
            if stale:
                archived_path = os.path.join(self.memory_dir, user_id, "l2", "archived.jsonl")
                with open(archived_path, "a", encoding="utf-8") as handle:
                    for entry in stale:
                        envelope = self._cipher.encrypt_json(
                            entry, aad=f"memory-archive-v1:{user_id}"
                        )
                        handle.write(json.dumps(envelope) + "\n")
            return {"archived": len(stale), "remaining": len(kept)}

        return {"error": f"未知命令: {command}"}

    def _write_l2(self, user_id: str, entries: list[dict]) -> None:
        """Explicit compatibility import/export helper; runtime never calls it."""
        current = {item["id"]: item for item in self._load_l2(user_id)}
        incoming = {str(item["id"]): dict(item) for item in entries}
        mutations = [("delete", item) for key, item in current.items() if key not in incoming]
        mutations.extend(("upsert", item) for item in incoming.values()
                         if current.get(item["id"]) != item)
        if mutations:
            self._commit_mutations(user_id, mutations)
        path = os.path.join(self.memory_dir, user_id, "l2", "memory.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for entry in entries:
                envelope = self._cipher.encrypt_json(
                    entry, aad=f"memory-l2-v1:{user_id}"
                )
                handle.write(json.dumps(envelope) + "\n")

    def _vector_delete(self, entry_id: str, user_id: str, *, strict: bool = False) -> None:
        collection = self._ensure_collection()
        if collection is None:
            if strict and self.embedder is not None:
                raise RuntimeError("memory vector collection unavailable")
            return
        try:
            collection.delete(ids=[entry_id])
        except Exception:
            if strict:
                raise

    # ============ 存储 ============
    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def _save_session(self, session: dict) -> None:
        session_id = str(session["session_id"])
        user_id = str(session["user_id"])
        expected = int(session.get("storage_revision", 0))
        revision = expected + 1
        payload = dict(session, storage_revision=revision)
        envelope = self._cipher.encrypt_json(
            payload, aad=f"memory-session-v2:{user_id}:{session_id}:{revision}"
        )
        with self._memory_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT user_id,revision FROM memory_session WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                if expected != 0:
                    raise RuntimeError("memory_session_revision_conflict")
                conn.execute("INSERT INTO memory_session VALUES(?,?,?,?,?,?)",
                             (session_id, user_id, revision, int(bool(session.get("archived"))), envelope, _now_iso()))
            else:
                if row[0] != user_id or row[1] != expected:
                    raise RuntimeError("memory_session_revision_conflict")
                conn.execute(
                    "UPDATE memory_session SET revision=?,archived=?,payload_json=?,updated_ts=? WHERE session_id=?",
                    (revision, int(bool(session.get("archived"))), envelope, _now_iso(), session_id),
                )
        session["storage_revision"] = revision

    def _load_session(self, session_id: str) -> dict | None:
        return self._read_session_row(session_id, archived=False)

    def _load_archived_session(self, session_id: str) -> dict | None:
        return self._read_session_row(session_id, archived=True)

    def _read_session_row(self, session_id: str, *, archived: bool) -> dict | None:
        with self._memory_connect() as conn:
            row = conn.execute(
                "SELECT user_id,revision,payload_json FROM memory_session WHERE session_id=? AND archived=?",
                (session_id, int(archived)),
            ).fetchone()
        if row is None:
            return None
        value = self._cipher.decrypt_json(row[2], aad=f"memory-session-v2:{row[0]}:{session_id}:{row[1]}")
        if (not isinstance(value, dict) or value.get("session_id") != session_id
                or value.get("user_id") != row[0] or value.get("storage_revision") != row[1]):
            raise ValueError("memory_session_identity_mismatch")
        return value

    def _load_session_path(self, path: str, session_id: str) -> dict | None:
        value = _load_json(path, None)
        if isinstance(value, str):
            decoded = self._cipher.decrypt_json(
                value, aad=f"memory-session-v1:{session_id}"
            )
            return decoded if isinstance(decoded, dict) else None
        return value if isinstance(value, dict) else None


def _guess_entity(content: str) -> str:
    """从事实内容猜实体: 优先英数词/专名, 兜底截断"""
    import re

    match = re.search(r"[A-Za-z][A-Za-z0-9_-]{1,20}", content)
    if match:
        return match.group(0)
    return content[:12]


def _type_to_relation(fact_type: str) -> str:
    return {
        "preference": "prefers",
        "task": "works_on",
        "constraint": "constraint",
        "fact": "has_property",
        "raw": "unknown",
    }.get(fact_type, "unknown")
