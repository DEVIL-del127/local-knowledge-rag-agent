# STUDY RAG V1-A v2 发布说明

发布日期：2026-09-21

Git 标签：`v2`

包版本：`2.0.0`

基线：`v1-a-final` / `169e8a4`

## 发布内容

本版本冻结 V1-A 的统一路由与业务链路。它保留现有 Elasticsearch/Chroma 检索、知识库
Generation、LangGraph、预算、幂等、取消和显式记忆机制，并增加：

- 后端权威 conversation/request 标识和会话隔离；
- 统一 RoutingService、上下文适用性和一次语义编码预算；
- 不可变 bge-m3 路由矩阵、类别阈值/间隔和超时熔断；
- AcceptedIR 单一语义来源、SourceSpan 审计和证据失败关闭；
- accepted task、路由依赖与 needs_revalidation 状态；
- 统一业务结果和生产装配验收。

## 最终验收证据

| 项目 | 结果 |
|---|---:|
| 未触碰 v4 封存表达 | 12/12 |
| 生产真实回答烟测 | 8/8 |
| 真实 ES/Chroma 固定集 | 74/80 |
| 其余结果 | 6 个诚实 `insufficient_evidence` |
| 端到端 P50 / P95 | 502.471 / 716.488 ms |
| 引用来源与已知页码准确率 | 100% |
| 伪造页码 | 0 |
| V1-A 聚焦回归 | 119 passed |

74/80 达到方案规定的 92.5% 阈值。六个未回答用例缺少逐文档、逐比较维度证据，系统没有
伪造比较结论。

## 范围边界

- 本版本不包含 V1-B 动态学习、候选审核或生产热发布。
- 本地 `.env`、私人 PDF、向量数据库、运行数据库、模型权重和日志不在发布包中。
- Elasticsearch、Ollama/bge-m3 和外部回答模型仍需在部署环境单独配置。
