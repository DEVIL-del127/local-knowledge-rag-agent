from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum

from .expression_service import ExpressionService


class RequestDomain(str, Enum):
    KB_DOCUMENT = "kb_document"
    GENERAL_CHAT = "general_chat"
    UTILITY_CALCULATE = "utility_calculate"
    MEMORY = "memory"
    HELP = "help"
    UNSUPPORTED = "unsupported"
    CLARIFY_DOMAIN = "clarify_domain"


@dataclass(frozen=True, slots=True)
class DomainDecision:
    domain: RequestDomain
    confidence: float
    reason_code: str
    router_version: str = "domain-router-v1"
    schema_version: str = "domain-decision-v1"

    def digest(self) -> str:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        payload["router_digest"] = self.digest()
        return payload


_HELP = re.compile(r"(?:^|\s)(?:help|帮助)(?:\s|$)|你能做什么|^(?:怎么使用|如何使用)|你是谁|有什么能力|有哪些功能|使用说明", re.I)
_MEMORY = re.compile(r"记住|保存记忆|保存.*偏好|修正记忆|记得我|还记得|我的偏好|查看记忆|回忆.*(?:名字|偏好)|记忆管理|删除记忆|清空记忆|忘掉|忘记", re.I)
_DAILY = re.compile(
    r"你好|您好|早上好|下午好|晚上好|谢谢|多谢|再见|吃啥|吃什么|午饭|晚饭|早餐|心情|"
    r"聊聊天|无聊|天气怎么样|讲个笑话|hello|hi\b|thanks?\b",
    re.I,
)
_KB = re.compile(
    r"论文|文献|文档|综述|知识库|论文库|文献库|\bpapers?\b|\bpublications?\b|pdf|doi|作者|期刊|会议|"
    r"检索|查资料|索引|入库|页码|研究|算法|模型|原理|作用|方法|实验|数据|字段|指标|筛选|查询|"
    r"额定转速|温度|网络|分析|总结|对比|比较|资料",
    re.I,
)
_TECHNICAL_TERM_QUERY = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9+.#_-]{1,24}"
    r"(?![A-Za-z0-9_]).{0,12}(?:是什么|是啥|作用|原理|怎么用|如何使用|如何训练|有哪些参数|适合什么任务|如何做)|"
    r"(?<![A-Za-z])[A-Z][A-Z0-9_-]{1,11}(?![A-Za-z])",
)
_UNSUPPORTED = re.compile(
    r"(?:删除|清空|修改|覆盖|重建|关闭|撤销).*(?:索引|数据库|记录|文件|文档|路由包|服务器|审计)|"
    r"(?:执行|运行).*(?:shell|命令|脚本)|绕过.*权限|导出.*(?:其他|别的)用户",
    re.I,
)
_GENERAL_TASK = re.compile(
    r"翻译|写一(?:段|封|篇|首|句|条)|润色|改写|讲.*笑话|讲个故事|"
    r"解释什么是|给.*(?:鼓励|建议|祝福|问候)|拟.*标题|介绍一下|"
    r"(?:语气|措辞).*(?:正式|礼貌)|改得.*礼貌|调整.*(?:表达|语气).*(?:客气|礼貌)|"
    r"用一句话解释|简要说明|一般的.*方法|感谢语",
    re.I,
)
_EXPLICIT_DOCUMENT = re.compile(r"知识库|论文|文献|文档|pdf|doi", re.I)
_CONTEXTUAL_FOLLOWUP = re.compile(
    r"那.{0,40}呢|"
    r"(?:19|20)\d{2}年?(?:的)?呢|"
    r"只看|仅看|其中|这些|上述|前(?:两|2|二)篇|换成|改成|年份|英文|中文|"
    r"(?:总结|归纳|对比|比较)(?:一下)?",
    re.I,
)
_BARE_CONTEXT = re.compile(
    r"^(?:继续(?:一下)?|看看这个|分析一下|查一下|总结一下|对比一下|比较一下|"
    r"分析它|评估一下它|那一个呢|还有吗|换一个|详细点|简单点|下一步呢|按之前的做|找相关的)$",
    re.I,
)


class DomainRouter:
    """Deterministic applicability router; it never interprets KB semantics."""

    def __init__(self, expression_service: ExpressionService | None = None) -> None:
        self.expression_service = expression_service or ExpressionService()

    def route(
        self,
        query: str,
        *,
        prior_domain: str | None = None,
        literature_reference: str | None = None,
    ) -> DomainDecision:
        value = str(query or "").strip()
        if not value:
            return DomainDecision(RequestDomain.CLARIFY_DOMAIN, 1.0, "empty_input")
        if self.expression_service.can_handle(value):
            return DomainDecision(RequestDomain.UTILITY_CALCULATE, 1.0, "bounded_expression")
        if _HELP.search(value):
            return DomainDecision(RequestDomain.HELP, 0.99, "explicit_help")
        if _MEMORY.search(value):
            return DomainDecision(RequestDomain.MEMORY, 0.98, "explicit_memory")
        if _UNSUPPORTED.search(value):
            return DomainDecision(RequestDomain.UNSUPPORTED, 0.99, "unsafe_mutation_request")
        if _GENERAL_TASK.search(value) and not _EXPLICIT_DOCUMENT.search(value):
            return DomainDecision(RequestDomain.GENERAL_CHAT, 0.99, "explicit_general_task")
        if (prior_domain == RequestDomain.KB_DOCUMENT.value
                and literature_reference not in {None, "", "none", "explicit_document"}):
            return DomainDecision(RequestDomain.KB_DOCUMENT, 0.99, "kb_literature_reference")
        if prior_domain == RequestDomain.KB_DOCUMENT.value and _CONTEXTUAL_FOLLOWUP.search(value):
            return DomainDecision(RequestDomain.KB_DOCUMENT, 0.96, "kb_context_followup")
        if prior_domain != RequestDomain.KB_DOCUMENT.value and _BARE_CONTEXT.fullmatch(value):
            return DomainDecision(RequestDomain.CLARIFY_DOMAIN, 0.99, "context_required")
        if _KB.search(value):
            return DomainDecision(RequestDomain.KB_DOCUMENT, 0.95, "document_or_technical_signal")
        if _DAILY.search(value):
            return DomainDecision(RequestDomain.GENERAL_CHAT, 0.98, "daily_conversation")
        if _TECHNICAL_TERM_QUERY.search(value):
            return DomainDecision(RequestDomain.KB_DOCUMENT, 0.90, "generic_technical_term_query")
        if len(value) <= 2:
            return DomainDecision(RequestDomain.CLARIFY_DOMAIN, 0.70, "short_ambiguous_input")
        return DomainDecision(RequestDomain.CLARIFY_DOMAIN, 0.0, "unknown_domain")
