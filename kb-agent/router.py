# -*- coding: utf-8 -*-
"""级联路由：nlu 规则层 → L3 embedding kNN → L4 历史日志投票 → 兜底

规则判定（L0）已抽象到 nlu.py（纯函数、表驱动、可单测可查看）。
本模块只负责：级联调度 + 模型层（kNN/历史）+ RouteResult 输出。
"""
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from es_client import ESClient
from embedder import Embedder
from nlu import analyze as nlu_analyze, StructuredQuery
from session_state import SessionState, apply_context
from config import (
    INDEX_EXAMPLES, INDEX_LOGS, KNN_SIM_HIGH, KNN_SIM_LOW, KNN_SIM_HIST,
    L0_ACCEPT_CONF, MAX_MAIN_SUBQUERIES,
)


@dataclass
class Entities:
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    year_exact: Optional[int] = None
    year_op: Optional[str] = None
    author: Optional[str] = None
    venue: Optional[str] = None
    language: Optional[str] = None
    doc_type: Optional[str] = None
    doc_ids: List[str] = field(default_factory=list)
    exclude: Dict = field(default_factory=dict)
    sort: Optional[str] = None
    stats: bool = False
    page: Optional[int] = None
    granularity: Optional[str] = None    # 解析粒度（year/month/day/hour/minute）；检索层按此降级
    time_ranges: List[Dict] = field(default_factory=list)  # 完整时间段；保留分钟粒度和时区

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], {}, False)}


@dataclass
class RouteResult:
    intent: str
    confidence: float
    source: str                 # rule_<name> / L3 / L4 / fallback / invalid / clarification / non_kb
    entities: Entities
    clarification_needed: bool = False
    clarification_reason: str = ""
    raw_query: str = ""

    def to_dict(self):
        return {
            "intent": self.intent, "confidence": round(self.confidence, 3),
            "source": self.source, "entities": self.entities.to_dict(),
            "clarification_needed": self.clarification_needed,
            "clarification_reason": self.clarification_reason,
        }


@dataclass
class RoutedSubQuery:
    """复合问题中的一个执行单元；非 main 角色只保留结构，不单独路由。"""
    index: int
    text: str
    role: str
    route: Optional[RouteResult] = None
    effective_query: str = ""
    time: Dict = field(default_factory=dict)
    time_ranges: List[Dict] = field(default_factory=list)
    entities: Dict = field(default_factory=dict)
    inherited: Dict = field(default_factory=dict)
    inherit_from: Optional[str] = None
    step_type: str = "retrieval"
    depends_on: List[int] = field(default_factory=list)
    requires_tools: List[str] = field(default_factory=list)
    executable: bool = True
    blocked_reason: str = ""
    parameters: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "index": self.index, "text": self.text, "role": self.role,
            "effective_query": self.effective_query,
            "time": self.time, "time_ranges": self.time_ranges,
            "entities": self.entities, "inherited": self.inherited,
            "inherit_from": self.inherit_from,
            "step_type": self.step_type, "depends_on": self.depends_on,
            "requires_tools": self.requires_tools, "executable": self.executable,
            "blocked_reason": self.blocked_reason,
            "parameters": self.parameters,
            "route": self.route.to_dict() if self.route else None,
        }


def sq_to_entities(sq: StructuredQuery) -> Entities:
    """nlu 结构化结果 → search 层 Entities（兼容层）
    非 year 粒度（month/day/hour/minute）降级为年份过滤 + granularity 标注（检索层按此降级）"""
    e = sq.entities
    t = sq.time

    def y(v):
        return int(str(v)[:4]) if v else None

    range_years = []
    for time_range in sq.time_ranges:
        for key in ("from", "to", "exact"):
            value = y(time_range.get(key))
            if value is not None:
                range_years.append(value)
    multi_year = len(set(range_years)) > 1

    return Entities(
        year_from=min(range_years) if multi_year else y(t.get("from")),
        year_to=max(range_years) if multi_year else y(t.get("to")),
        year_exact=None if multi_year else y(t.get("exact")),
        year_op="between" if multi_year else t.get("op"),
        author=e.get("author"), venue=e.get("venue"),
        language=e.get("language"), doc_type=e.get("doc_type"),
        doc_ids=e.get("doc_ids") or [], exclude=e.get("exclude") or {},
        sort=t.get("sort"), stats=False,
        granularity=t.get("granularity"),
        time_ranges=sq.time_ranges or ([dict(t)] if t else []),
    )


class Router:
    def __init__(self, es: ESClient = None, emb: Embedder = None,
                 state: SessionState = None):
        self.es = es or ESClient()
        self.emb = emb or Embedder()
        self.state = state or SessionState()
        self._llm = None          # 懒加载 LLM 意图分类器
        self._llm_failed = False  # 熔断：LLM 不可用则不再尝试

    def _llm_intent(self, q: str):
        """L2 LLM 意图分类（熔断保护：失败一次就不再调；KB_NO_LLM=1 环境变量可关）"""
        if os.environ.get("KB_NO_LLM") == "1":
            return None
        if self._llm_failed:
            return None
        from llm_intent import need_llm_intent
        if not need_llm_intent(q):
            return None
        if self._llm is None:
            try:
                from llm_intent import LLMIntentClassifier
                self._llm = LLMIntentClassifier()
            except Exception as e:
                self._llm_failed = True
                return None
        try:
            return self._llm.classify(q)
        except Exception:
            self._llm_failed = True
            return None

    def _update_state(self, query: str, res: RouteResult):
        """路由后更新会话状态（供 L3 上下文意图使用）"""
        self.state.update(res.intent, res.entities.to_dict() if res.entities else {},
                          query)

    # ================= L3 / L4（模型层） =================
    def knn_layer(self, q: str, ent: Entities) -> Optional[RouteResult]:
        qv = self.emb.encode([q], query_mode=True)[0]
        hits = self.es.hits(self.es.knn(INDEX_EXAMPLES, qv.tolist(), k=1, source=["intent", "text"]))
        if not hits:
            return None
        sim, intent = hits[0]["_score"], hits[0]["_source"]["intent"]
        if sim >= KNN_SIM_HIGH:
            return RouteResult(intent, sim, "L3", ent, raw_query=q)
        if sim >= KNN_SIM_LOW:
            # 低置信区（0.85~0.95）：不直接采纳，标记 L3_low 供 L2 LLM 决策
            return RouteResult(intent, sim, "L3_low", ent, raw_query=q)
        return None

    def history_layer(self, qv) -> Optional[RouteResult]:
        hits = self.es.hits(self.es.knn(INDEX_LOGS, qv.tolist(), k=3, field="query_embedding",
                                        source=["intent", "feedback", "confidence"]))
        votes, total = {}, 0.0
        for h in hits:
            if h["_score"] < KNN_SIM_HIST:
                continue
            if h["_source"].get("feedback") == "wrong":
                continue
            intent = h["_source"]["intent"]
            votes[intent] = votes.get(intent, 0) + 1
            total += h["_score"]
        if not votes:
            return None
        best = max(votes, key=votes.get)
        return RouteResult(best, total / len(hits) if hits else 0.4, "L4", Entities(), raw_query="")

    # ================= 总入口 =================
    def route(self, query: str, session: Dict = None) -> RouteResult:
        sq = nlu_analyze(query, session)

        # 规则层命中：确定性规则直接返回；宽泛规则保留为候选，继续让模型复核。
        rule_candidate = None
        if sq.intent:
            ent = sq_to_entities(sq)
            rule_candidate = RouteResult(
                sq.intent, sq.confidence, sq.source, ent,
                clarification_needed=(sq.intent == "clarification"),
                clarification_reason=sq.clarification,
                raw_query=query,
            )
            terminal = sq.intent in {"invalid", "non_kb", "clarification"}
            if terminal or sq.confidence >= L0_ACCEPT_CONF:
                self._update_state(query, rule_candidate)
                return rule_candidate

        # 上下文层（L3）：短句/指代 → 继承上一意图（规则未命中时优先于模型层）
        ctx = apply_context(sq.query, self.state)
        if ctx:
            intent, entities, src, reason = ctx
            res = RouteResult(intent, 0.7, src, Entities(**entities), raw_query=query)
            self._update_state(query, res)
            return res

        # 规则层未命中 → L3（kNN）
        ent = sq_to_entities(sq)
        try:
            res = self.knn_layer(sq.query, ent)
        except Exception:
            # ES/Ollama 暂时不可用时仍可使用规则候选，不让完整路由整体失败。
            res = None
        if res:
            # L3_low（0.85~0.95 低置信区）→ L2 LLM 意图分类细判
            if res.source == "L3_low":
                llm_r = self._llm_intent(sq.query)
                if llm_r is not None:
                    # 一致性门槛：LLM 与 kNN 不一致时要求更高置信（防小模型误判覆盖）
                    need_conf = 0.85 if llm_r.intent == res.intent else 0.90
                    if llm_r.confidence >= need_conf:
                        res = RouteResult(llm_r.intent, llm_r.confidence, "L2_llm",
                                          ent, raw_query=query)
            # 低置信 kNN 与更可靠的规则候选冲突时不强行覆盖。
            if (res.source == "L3_low" and rule_candidate is not None
                    and rule_candidate.confidence >= res.confidence):
                rule_candidate.source = f"{rule_candidate.source}_candidate_fallback"
                res = rule_candidate
            self._update_state(query, res)
            return res

        if rule_candidate is not None:
            rule_candidate.source = f"{rule_candidate.source}_candidate_fallback"
            self._update_state(query, rule_candidate)
            return rule_candidate

        # 兜底：语义检索 + 完整性检查
        res = RouteResult("semantic_retrieval", 0.3, "fallback", ent, raw_query=query)
        self._update_state(query, res)
        return res

    def route_many(self, query: str, session: Dict = None) -> List[RoutedSubQuery]:
        """先拆复合问题，再仅对主问题逐一执行完整意图路由。

        时间/约束/输出子句不单独调用模型；继承的时间原文会加入 effective_query，
        让 L0/kNN 看到完整语境，同时结构化字段会覆盖回 RouteResult，供检索层使用。
        """
        from cleaner import clean as clean_text
        from splitter import split

        cleaned = clean_text(query).cleaned
        subs = split(cleaned, mode="rule", analyze_fn=nlu_analyze)
        if not subs:
            return []

        main_count = sum(sub.role == "main" for sub in subs)
        if main_count > MAX_MAIN_SUBQUERIES:
            reason = (f"一次识别到 {main_count} 个主问题，超过上限 {MAX_MAIN_SUBQUERIES}；"
                      "请分批提问，避免并发模型调用和检索扇出")
            route = RouteResult(
                "clarification", 1.0, "clarification_subquery_limit", Entities(),
                clarification_needed=True, clarification_reason=reason, raw_query=query,
            )
            return [RoutedSubQuery(1, cleaned, "main", route=route,
                                   effective_query=cleaned)]

        # 极端情况下角色分类没有 main，保持旧行为：整句作为一个主问题路由。
        if not any(sub.role == "main" for sub in subs):
            route = self.route(cleaned, session)
            return [RoutedSubQuery(1, cleaned, "main", route=route,
                                   effective_query=cleaned)]

        planned = []
        for index, sub in enumerate(subs, 1):
            item = RoutedSubQuery(
                index=index, text=sub.text, role=sub.role,
                time=dict(sub.time), time_ranges=[dict(t) for t in sub.time_ranges],
                entities=dict(sub.entities), inherited=dict(sub.inherited),
                inherit_from=sub.inherit_from,
                step_type=sub.step_type, depends_on=list(sub.depends_on),
                requires_tools=list(sub.requires_tools), executable=sub.executable,
                blocked_reason=sub.blocked_reason,
                parameters=dict(sub.parameters),
            )
            if sub.role != "main":
                planned.append(item)
                continue

            inherited_raw = []
            for time_expr in sub.time_ranges:
                raw = time_expr.get("raw", "").strip()
                if raw and raw not in sub.text and raw not in inherited_raw:
                    inherited_raw.append(raw)
            item.effective_query = "，".join(inherited_raw + [sub.text])
            structured = StructuredQuery(
                query=sub.text, entities=dict(sub.entities), time=dict(sub.time),
                time_ranges=[dict(t) for t in sub.time_ranges],
            )
            if not item.executable and (item.requires_tools or item.depends_on):
                # 依赖型步骤只做确定性规划，不继续扇出 kNN/LLM 调用。
                intent = "numerical_calculation" if item.step_type == "calculation" else "agent_workflow"
                item.route = RouteResult(
                    intent, 0.99, "workflow_rule", sq_to_entities(structured),
                    raw_query=sub.text,
                )
            else:
                item.route = self.route(item.effective_query, session)

            # Router 已看过 effective_query；这里再覆盖 splitter 的继承快照，防止字段丢失。
            overlay = sq_to_entities(structured)
            for name in Entities.__dataclass_fields__:
                value = getattr(overlay, name)
                if value not in (None, [], {}, False):
                    setattr(item.route.entities, name, value)
            item.route.raw_query = sub.text
            planned.append(item)
        return planned


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    router = Router()
    tests = [
        "找25年之前的文献", "23年之前关于贝叶斯的文章", "2024年关于缺失值插补的论文",
        "张昭昭的双储层网络和他那篇回声信念网络有什么区别", "帮我写个Python脚本",
        "库里总共有多少篇文献", "这篇论文的方法是什么", "information-15-00222 这篇论文的方法是什么",
        "讲讲回声状态网络的训练方法", "2210.02040v3.pdf", "？", "去年发表的",
        "最新的文献有哪些", "2021年Neurocomputing那篇引用了哪些文献", "那篇NeurIPS论文的作者是谁",
    ]
    for t in tests:
        r = router.route(t)
        print(f"Q: {t}")
        print(f"  -> {r.intent:<22} conf={r.confidence:.3f} src={r.source} clar={r.clarification_reason or '-'}")
        if r.entities.to_dict():
            print(f"     entities: {r.entities.to_dict()}")
