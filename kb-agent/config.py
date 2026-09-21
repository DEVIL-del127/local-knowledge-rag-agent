# -*- coding: utf-8 -*-
"""kb-agent 配置"""
import os

ES_URL = os.environ.get("KB_ES_URL", "http://localhost:9200")
# kb-agent 的向量统一由本地 Ollama 提供，避免依赖 Hugging Face 缓存。
OLLAMA_URL = os.environ.get("KB_OLLAMA_URL", os.environ.get("OLLAMA_URL", "http://localhost:11434"))
MODEL_NAME = os.environ.get("KB_EMBED_MODEL", os.environ.get("EMBED_MODEL", "bge-m3"))
# 保留 query 侧任务描述，减少路由 query 与意图示例的表述差异。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
EMBED_DIM = 1024

INDEX_DOCS = "pdf_documents"
INDEX_CHUNKS = "pdf_chunks"
INDEX_EXAMPLES = "intent_examples"
INDEX_LOGS = "query_logs"

# 路由阈值（2026-08-21 leave-one-out 校准：172 种子，32 例 L1 职责）
# 相似度分布: min=0.797 p25=0.981 median=0.992 p75=0.995 max=0.997
# 结论：≥0.95 正确率 100%（语义同义表达成熟）；0.85~0.95 为低置信区 → L2 LLM 决策；<0.85 交给上下文/兜底
KNN_SIM_HIGH = 0.95   # kNN 置信采纳（≥0.95 直接信）
KNN_SIM_LOW = 0.85    # kNN 低置信下界（0.85~0.95 → 留给 L2 LLM 决策，不直接采纳）
KNN_SIM_HIST = 0.45   # L4 历史样本采纳阈值（历史投票语义弱，保持低）
L0_ACCEPT_CONF = 0.95 # L0 直接截断阈值；低于此值保留规则候选并继续模型链
MAX_MAIN_SUBQUERIES = 8  # 单次复合问题的模型/检索扇出上限
LLM_CONF = 0.6

# 分块
CHUNK_MAX_LEN = 512

# 知识库规模阈值：<= SMALL_KB 时属性类一律全枚举
SMALL_KB = 100

# 意图定义
INTENTS = [
    "semantic_retrieval", "attribute_filter", "inventory",
    "doc_qa", "cross_doc_synthesis", "citation_query",
    "metadata_query", "hybrid", "non_kb", "invalid", "clarification",
    "numerical_calculation", "agent_workflow",
]

INTENT_LABELS = {
    "semantic_retrieval": "I1 语义检索",
    "attribute_filter": "I2 属性筛选",
    "inventory": "I3 全库盘点",
    "doc_qa": "I4 单文档问答",
    "cross_doc_synthesis": "I5 跨文档对比",
    "citation_query": "I6 参考文献查询",
    "metadata_query": "I7 文档属性",
    "hybrid": "I8 混合(属性+语义)",
    "numerical_calculation": "I9 数值计算(增长率/差值/占比/倍数)",
    "agent_workflow": "I10 Agent 依赖工作流(等待 Planner/MCP)",
    "non_kb": "I11 非知识库任务",
    "invalid": "无效输入",
    "clarification": "需澄清",
}

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# LLM 结构解析配置（默认：本地 Ollama qwen）
LLM = {
    "provider": "openai",            # openai（Ollama 本地 DeepSeek，OpenAI 兼容）/ zhipu / mock / none
    "api_key": "ollama",             # Ollama 本地无需真实 key（占位）
    "base_url": "http://localhost:11434/v1",
    "model": "qwen2.5:3b",
    "conf_threshold": 0.6,           # 解析置信度门控，低于此降级规则
    "max_retry": 1,                  # JSON 解析失败重试次数
    "trigger": {"max_query_len": 80, "min_subqueries": 4},
}
