from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from contextlib import ExitStack

def _setup_logging() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        from telemetry.log_setup import check_log_disk_watermark, setup_app_logging
        setup_app_logging(log_dir=os.environ.get("APP_LOG_DIR", "logs/app"), console=True)
        check_log_disk_watermark()
    except Exception:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from agent.agent_cli import run_agent_loop
from agent.agent_limits import CircuitBreaker, PersistentCache, RateLimiter, TokenBudget
from agent.agent_service import AgentSettings, PrivateKnowledgeAgent
from agent.agent_skills import SkillRegistry
from agent.app_settings import AppSettings
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.model_call_ledger import ModelCallLedger
from memory.memory_manager import MemoryConfig, MemoryManager
from telemetry.search_log import SearchLogConfig, SearchRecorder

# 防护层默认值(可用环境变量覆盖)
DEFAULT_STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_state")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class AgentApplication:
    agent: object
    label: str
    user_id: str
    memory: object
    session_id: str
    recorder: object
    telemetry_bus: object
    closed: bool = False

    def run(self):
        try:
            run_agent_loop(self.agent, label=self.label, user_id=self.user_id,
                           memory=self.memory, session_id=self.session_id)
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.memory.on_session_end(session_id=self.session_id, user_id=self.user_id)
        finally:
            try:
                registry = getattr(self.agent, "registry", None)
                if registry is not None and registry.has("discover_documents"):
                    gateway = getattr(registry.get("discover_documents"), "gateway", None)
                    if gateway is not None:
                        _safe_shutdown(gateway.close)
            finally:
                try:
                    self.telemetry_bus.shutdown()
                finally:
                    self.recorder.shutdown()


def build_application(app_settings: AppSettings) -> AgentApplication:
    # Keep the original setup error even when a shutdown callback also fails.
    with ExitStack() as resources:
        application = _build_application(app_settings, resources)
        resources.pop_all()
        return application


def _safe_shutdown(callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except Exception:
        logger.warning("Resource cleanup failed", exc_info=True)


def _build_application(app_settings: AppSettings, resources: ExitStack) -> AgentApplication:
    project_root = str(app_settings.project_root)
    es_host = app_settings.es_host
    es_port = app_settings.es_port
    ollama_url = app_settings.ollama_url
    embed_model = app_settings.embed_model
    agent_db = app_settings.agent_db
    user_id = app_settings.user_id
    # The contract-first runtime is the production default.  Legacy behavior is
    # available only through an explicit environment override for rollback.
    execution_profile = app_settings.execution_profile
    if not os.environ.get("AGENT_EXECUTION_PROFILE", "").strip():
        if "AGENT_RUNTIME_MODE" in os.environ or "SEMANTIC_COMPILER_MODE" in os.environ:
            logger.warning(
                "AGENT_RUNTIME_MODE/SEMANTIC_COMPILER_MODE 已弃用；请改用 AGENT_EXECUTION_PROFILE"
            )
    runtime_mode = execution_profile
    semantic_mode = "shadow" if execution_profile == "shadow" else "off"

    state_dir = str(app_settings.state_dir)
    os.makedirs(state_dir, exist_ok=True)

    from agent.lazy_backend import LazySearchBackend, LazyDependency
    def backend_factory():
        from main import PDFSearchSystem
        return PDFSearchSystem(es_host=es_host, es_port=es_port, db=agent_db,
                               ollama_url=ollama_url, embed_model=embed_model,
                               read_only=execution_profile == "enforce", project_root=project_root)
    if agent_db not in {"main", "test"}:
        raise ValueError("unsupported AGENT_DB")
    system = LazySearchBackend(
        backend_factory, label="正式" if agent_db == "main" else "测试",
        pdf_dir=os.path.join(project_root, "pdfs" if agent_db == "main" else "tests/pdfs"),
        index_name="pdf_documents" if agent_db == "main" else "pdf_documents_test",
    )

    # ---------- 防护层装配 ----------
    token_budget = TokenBudget(
        os.path.join(state_dir, "token_usage.json"),
        session_limit=_env_int("AGENT_SESSION_TOKEN_BUDGET", 100_000),
        daily_limit=_env_int("AGENT_DAILY_TOKEN_BUDGET", 500_000),
    )
    rate_limiter = RateLimiter(
        os.path.join(state_dir, "rate_limit.sqlite3"),
        max_calls=_env_int("AGENT_RATE_LIMIT_PER_MIN", 20),
        window_seconds=60,
    )
    circuit_breaker = CircuitBreaker(
        failure_threshold=_env_int("AGENT_BREAKER_THRESHOLD", 3),
        cooldown_seconds=_env_int("AGENT_BREAKER_COOLDOWN", 60),
    )
    answer_cache = PersistentCache(
        os.path.join(state_dir, "answer_cache.sqlite3"),
        max_entries=64,
        ttl_seconds=_env_int("AGENT_CACHE_TTL_HOURS", 24) * 3600,
    )

    # 检索词记录(旁路, 独立于 Agent 主流程)
    search_log_config = SearchLogConfig.from_env()
    recorder = SearchRecorder.from_config(search_log_config)
    resources.callback(_safe_shutdown, recorder.shutdown)

    # v2: TelemetryBus(request context → search_log + trace 双流)
    from telemetry.bus import TelemetryBus

    telemetry_bus = TelemetryBus(
        node_id=search_log_config.node_id,
        config=search_log_config,
        recorder=recorder,
    )

    deepseek_client = DeepSeekClient(
        DeepSeekSettings.from_env(),
        budget=token_budget,
        breaker=circuit_breaker,
        ledger=ModelCallLedger(os.path.join(state_dir, "model_calls.sqlite3")),
    )
    resources.callback(_safe_shutdown, telemetry_bus.shutdown)

    # 记忆系统(M1+M2: 写入链路 + 召回 + 管理)
    memory = MemoryManager(
        llm_client=deepseek_client,
        state_dir=state_dir,
        config=MemoryConfig(),
        embedder=system.embedder,  # 复用 bge-m3 做向量检索
    )
    session_id = memory.on_session_start(user_id=user_id)
    resources.callback(_safe_shutdown, memory.abort_session, session_id=session_id, user_id=user_id)
    logger.info("记忆会话已创建: %s", session_id)

    generation_registry = None
    skill_registry = SkillRegistry()
    if runtime_mode in {"shadow", "enforce"}:
        from ingestion.pipeline.generation_registry import GenerationRegistry

        generation_registry = LazyDependency(lambda: GenerationRegistry(app_settings.ingestion_registry_path, read_only=True))
        try:
            from agent.mcp.config import build_clients, load_server_configs
            from agent.mcp.skills import McpInternalSkill

            mcp_config = str(app_settings.mcp_config_path)
            mcp_clients = build_clients(mcp_config)
            configs = {item.name: item for item in load_server_configs(mcp_config)}
            for server_name, client in mcp_clients.items():
                config = configs[server_name]
                for tool_name in sorted(config.allowed_tools):
                    skill_registry.register(McpInternalSkill(
                        server_name, tool_name, client, read_only=config.read_only
                    ))
        except Exception as exc:
            if runtime_mode == "enforce":
                logger.warning("MCP 配置未就绪；MCP skills 已禁用: %s", exc)
            else:
                logger.debug("MCP 配置未启用: %s", exc)
    semantic_compiler = None
    if semantic_mode != "off" or runtime_mode in {"shadow", "enforce"}:
        from agent.semantic_compiler_adapter import SemanticCompilerAdapter
        from core.contract_store import ContractStore

        semantic_compiler = SemanticCompilerAdapter(
            search_backend=system,
            generation_registry=generation_registry,
            enable_llm=app_settings.semantic_compiler_llm and runtime_mode != "shadow",
            require_active_generation=runtime_mode == "enforce",
            contract_store=ContractStore(app_settings.ingestion_registry_path.parent / "contracts",
                                         store_id="local-contracts-v1"),
        )

    def _legacy_factory():
        return PrivateKnowledgeAgent(
            search_backend=system,
            registry=skill_registry,
            deepseek_client=deepseek_client,
            settings=AgentSettings(
                state_dir=state_dir,
                semantic_compiler_mode=semantic_mode,
            ),
            cache=answer_cache,
            rate_limiter=rate_limiter,
            recorder=recorder,
            memory=memory,
            telemetry_bus=telemetry_bus,
            semantic_compiler=semantic_compiler,
            generation_registry=generation_registry,
        )

    if runtime_mode in {"shadow", "enforce"}:
        from agent.runtime.checkpoint_store import SQLiteCheckpointStore
        from agent.runtime.clarification_presenter import LLMClarificationPresenter
        from agent.runtime.coordinator import RuntimeCoordinator
        from agent.runtime.facade import RuntimeAgentFacade
        from agent.runtime.graph import RuntimeSemanticGraph
        from agent.routing_service import RoutingService
        from agent.semantic_matrix import SemanticMatrixRouter
        from core.embedder import OllamaEmbedder

        semantic_router = None
        routing_package = app_settings.project_root / "data" / "routing" / "v1a" / "package.json"
        if routing_package.is_file():
            try:
                routing_embedder = OllamaEmbedder(
                    base_url=app_settings.ollama_url, model=app_settings.embed_model, timeout=30,
                )
                semantic_router = SemanticMatrixRouter(
                    routing_package, routing_embedder.embed_query, timeout_seconds=2.0,
                )
            except Exception:
                logger.warning("V1-A routing package unavailable; semantic auto-release remains disabled",
                               exc_info=True)

        coordinator = RuntimeCoordinator(
            compiler=semantic_compiler,
            checkpoint_store=SQLiteCheckpointStore(
                os.path.join(state_dir, "checkpoints", "shadow.sqlite3" if runtime_mode == "shadow" else "runtime.sqlite3")
            ),
            clarification_rewriter=(
                LLMClarificationPresenter(deepseek_client).rewrite
                if app_settings.llm_clarification else None
            ),
            graph=RuntimeSemanticGraph(semantic_compiler),
            routing_service=RoutingService(semantic_router=semantic_router),
        )
        agent = RuntimeAgentFacade(
            legacy=_legacy_factory(), mode=runtime_mode, coordinator=coordinator,
        )
    else:
        # Hard bypass: no runtime/checkpoint module is imported in off mode.
        agent = _legacy_factory()

    logger.info(
        "防护层就绪: 会话预算=%d/每日预算=%d | 限流=%d次/分钟 | 熔断阈值=%d/冷却=%ds | 缓存TTL=%dh | 检索词日志=%s | trace=%s",
        token_budget.session_limit, token_budget.daily_limit,
        rate_limiter.max_calls, circuit_breaker.failure_threshold,
        circuit_breaker.cooldown_seconds, answer_cache.ttl_seconds // 3600,
        search_log_config.json_dir,
        "telemetry_bus",
    )
    logger.info(
        "语义编译接入: mode=%s | NLU LLM=%s | RequestIR/LogicalPlan=authoritative",
        "enforce(runtime)" if runtime_mode == "enforce" else agent._semantic_compiler_mode,
        "开启" if getattr(semantic_compiler, "enable_llm", False) else "关闭",
    )
    logger.info("Agent runtime: mode=%s", runtime_mode)

    return AgentApplication(agent, system.label, user_id, memory, session_id, recorder, telemetry_bus)
