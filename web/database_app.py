from __future__ import annotations

import os
import socket
import sys
import threading
import time
import uuid
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationState
from main import PDFSearchSystem


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
REGISTRY_PATH = ROOT / "data" / "ingestion" / "generation_registry.sqlite3"
MAX_PDF_BYTES = 80 * 1024 * 1024

# nlu_v2 uses a source-layout package under kb-agent.  Make direct uvicorn
# launches behave the same as agent_main.py and the supported web launcher.
NLU_ROOT = ROOT / "kb-agent"
if str(NLU_ROOT) not in sys.path:
    sys.path.insert(0, str(NLU_ROOT))


@dataclass(slots=True)
class BuildJob:
    job_id: str
    database: str
    state: str = "queued"
    phase: str = "等待执行"
    progress: int = 2
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    generation_id: str = ""
    accepted_documents: int = 0
    quarantined_documents: int = 0
    chunk_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivateRequest(BaseModel):
    generation_id: str = Field(min_length=8, max_length=128)
    confirm_generation_id: str = Field(min_length=8, max_length=128)
    database: str = Field(default="main", pattern="^(main|test)$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    database: str = Field(default="main", pattern="^(main|test)$")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    conversation_id: str | None = Field(default=None, min_length=16, max_length=128)
    request_id: str | None = Field(default=None, min_length=16, max_length=128)


class ConversationRequest(BaseModel):
    database: str = Field(default="main", pattern="^(main|test)$")


class AgentPool:
    """Lazily owns one application per database for the local web process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._apps: dict[str, Any] = {}

    def get(self, database: str):
        with self._lock:
            if database not in self._apps:
                from agent.app_settings import AppSettings
                from agent.application import build_application
                previous = os.environ.get("AGENT_DB")
                os.environ["AGENT_DB"] = database
                try:
                    self._apps[database] = build_application(AppSettings.from_env(ROOT))
                finally:
                    if previous is None:
                        os.environ.pop("AGENT_DB", None)
                    else:
                        os.environ["AGENT_DB"] = previous
            return self._apps[database]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._jobs: dict[str, BuildJob] = {}

    def create(self, database: str) -> BuildJob:
        with self._lock:
            if any(job.state in {"queued", "running"} for job in self._jobs.values()):
                raise RuntimeError("已有建库任务正在执行，请等待其结束。")
            job = BuildJob(job_id=uuid.uuid4().hex, database=database)
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job.job_id,), daemon=True).start()
        return job

    def get(self, job_id: str) -> BuildJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> BuildJob | None:
        with self._lock:
            return max(self._jobs.values(), key=lambda job: job.created_at, default=None)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        with self._build_lock:
            try:
                self._update(job_id, state="running", phase="解析 PDF 并构建隔离索引",
                             progress=18, started_at=time.time())
                system = PDFSearchSystem(db=job.database, project_root=ROOT)
                manifest = system.process_pdfs()
                if not manifest:
                    raise RuntimeError("没有生成 PREPARED generation，请检查 PDF 与服务日志。")
                self._update(
                    job_id,
                    state="prepared",
                    phase="质量校验通过，等待人工激活",
                    progress=100,
                    finished_at=time.time(),
                    generation_id=str(manifest.get("generation_id") or ""),
                    accepted_documents=int(manifest.get("accepted_document_count") or 0),
                    quarantined_documents=int(manifest.get("quarantined_document_count") or 0),
                    chunk_count=int(manifest.get("vector_chunk_count") or 0),
                )
            except Exception as exc:
                self._update(job_id, state="failed", phase="建库失败", progress=100,
                             finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")


jobs = JobManager()
agents = AgentPool()
app = FastAPI(title="STUDY 文献数据库控制台", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: ConversationRequest) -> dict[str, str]:
    return {"conversation_id": uuid.uuid4().hex, "database": payload.database}


def _database_dir(database: str) -> Path:
    if database not in PDFSearchSystem.DB_CONFIG:
        raise HTTPException(status_code=422, detail="database 必须是 main 或 test")
    relative = PDFSearchSystem.DB_CONFIG[database]["pdf_dir"]
    target = (ROOT / relative).resolve()
    if ROOT.resolve() not in target.parents:
        raise HTTPException(status_code=500, detail="PDF 目录配置越界")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _registry() -> GenerationRegistry:
    return GenerationRegistry(REGISTRY_PATH)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/status")
def status(database: str = Query(default="main", pattern="^(main|test)$")) -> dict[str, Any]:
    source = _database_dir(database)
    pdfs = sorted(source.glob("*.pdf"))
    registry = _registry()
    active, revision = registry.get_active()
    generations = sorted(registry.list_all(), key=lambda item: item.generation_id, reverse=True)[:12]
    latest = jobs.latest()
    return {
        "database": database,
        "pdf_directory": str(source),
        "pdf_count": len(pdfs),
        "pdf_bytes": sum(item.stat().st_size for item in pdfs),
        "active_generation": active.to_dict() if active else None,
        "registry_revision": revision,
        "generations": [item.to_dict() for item in generations],
        "latest_job": latest.to_dict() if latest else None,
    }


@app.get("/api/pdfs")
def list_pdfs(database: str = Query(default="main", pattern="^(main|test)$")) -> dict[str, Any]:
    files = sorted(_database_dir(database).glob("*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)
    return {"database": database, "files": [
        {"name": item.name, "bytes": item.stat().st_size, "modified_at": item.stat().st_mtime}
        for item in files
    ]}


@app.delete("/api/pdfs/{filename}")
def delete_pdf(filename: str, database: str = Query(default="main", pattern="^(main|test)$")) -> dict[str, Any]:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.casefold().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="文件名无效")
    target = _database_dir(database) / safe_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    target.unlink()
    return {"deleted": safe_name, "database": database}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    history = [
        {"role": item.get("role", ""), "content": item.get("content", "")[:12000]}
        for item in payload.history[-12:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    try:
        if not payload.conversation_id or not payload.request_id:
            raise HTTPException(status_code=422, detail="conversation_id 和 request_id 由客户端显式携带")
        application = agents.get(payload.database)
        reply = application.agent.chat(payload.message.strip(), history=history,
                                       user_id=application.user_id,
                                       session_id=payload.conversation_id,
                                       request_id=payload.request_id)
        return {**reply.to_dict(), "conversation_id": payload.conversation_id,
                "request_id": payload.request_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"问答服务暂不可用：{type(exc).__name__}: {exc}") from exc


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _url_open(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return _port_open(parsed.hostname or "localhost", parsed.port or 11434)


@app.get("/api/health")
def health() -> dict[str, Any]:
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    return {
        "api": True,
        "elasticsearch": _port_open(os.environ.get("ES_HOST", "localhost"), int(os.environ.get("ES_PORT", "9200"))),
        "ollama": _url_open(ollama_url),
        "ollama_url": ollama_url,
        "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "embed_model": os.environ.get("EMBED_MODEL", "bge-m3"),
    }


@app.post("/api/pdfs")
async def upload_pdf(
    request: Request,
    filename: str = Query(min_length=5, max_length=220),
    database: str = Query(default="main", pattern="^(main|test)$"),
) -> dict[str, Any]:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.casefold().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="仅允许无路径的 .pdf 文件名")
    body = await request.body()
    if not body or len(body) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF 为空或超过 80 MiB")
    if not body.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="文件签名不是 PDF")
    target = _database_dir(database) / safe_name
    target.write_bytes(body)
    return {"filename": safe_name, "bytes": len(body), "database": database}


@app.post("/api/build", status_code=202)
def start_build(database: str = Query(default="main", pattern="^(main|test)$")) -> dict[str, Any]:
    source = _database_dir(database)
    if not any(source.glob("*.pdf")):
        raise HTTPException(status_code=409, detail="PDF 目录为空，请先上传文件。")
    try:
        return jobs.create(database).to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.to_dict()


@app.post("/api/activate")
def activate(payload: ActivateRequest) -> dict[str, Any]:
    if payload.generation_id != payload.confirm_generation_id:
        raise HTTPException(status_code=409, detail="确认 generation ID 不一致")
    registry = _registry()
    record = registry.get(payload.generation_id)
    if record is None or record.state not in {GenerationState.PREPARED, GenerationState.RETIRED}:
        raise HTTPException(status_code=409, detail="generation 不存在或当前不可激活")
    try:
        revision = PDFSearchSystem(db=payload.database, project_root=ROOT).activate_generation(
            payload.generation_id
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"激活失败：{type(exc).__name__}: {exc}") from exc
    return {"generation_id": payload.generation_id, "state": "active", "registry_revision": revision}
