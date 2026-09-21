# V1-A implementation and gate report

Date: 2026-09-20

## Implemented

- Backend-owned conversation and request identities for Web chat.
- Structured `RoutingHint` and `ContextProposal` contracts with business/general-chat/unknown applicability.
- Context validity check before domain termination; revalidation-required state cannot provide executable objects.
- Fixed bge-m3/NumPy matrix package, same preprocessing for examples and queries, normalized-vector and digest checks.
- Per-category top-three distinct verified expression-cluster mean, threshold and margin checks.
- Context-dependent expressions excluded by the package builder.
- Two-second caller timeout, explicit inflight/timeout-pending state, and fail-closed circuit state.
- Explicit finished/cancel-confirmed audit states, late-completion accounting, timeout/callback race protection, 30-second half-open single probe, and health-confirmed recovery.
- Independent SourceSpanAudit for actions, negation, years, quantities and object expressions.
- Removed the compiler adapter's unconditional executable override; unresolved validation remains authoritative.
- Accepted task/dependency/revalidation lifecycle is persisted across turns and cleared on non-KB transitions.
- Empty results clear the current result pointer; later ordinal references cannot reuse an older list.
- Context proposals now cover bound comparison, continuation, field patch, missing-object and conflicting-candidate exits.
- Runtime consumes the compiler's literature projection instead of re-running full literature parsing in routing and preparation.
- Admission and pending-clarification screening use deterministic routing; the authoritative route consumes at most one semantic encoding per new turn, while replay consumes none.
- Checked-in acceptance tooling validates the exact 200-case stratum contract, compares baseline/candidate hard gates, reports Wilson intervals, evaluates per-category evidence, and records all three latency classes.

## Category switches

Following explicit user authorization, `literature_find`, `literature_compare`, and `general_chat` are enabled in the rebuilt immutable package. The package contains 35 expression clusters and retains per-category threshold and margin checks.

## Performance probe

Hardware/environment: current local Windows host, Ollama `bge-m3`, nine-example fixed matrix. Measurements are development probes, not a production reliability claim.

| Condition | Result |
|---|---:|
| Cold package load plus first query | 305.760 ms |
| Warm idle P50, 10 queries | 31.135 ms |
| Warm idle P95, 10 queries | 1703.281 ms |
| Warm idle maximum | 1986.283 ms |
| One query during 32-item background embedding | 227.763 ms |

The warm P95 target of at most 1 second was not met in this bounded probe. Per the V1 plan, no semantic category is enabled and the verified deterministic route remains active. A larger controlled benchmark is required before reconsidering switches.

### Re-measurement with persistent HTTP session

- OllamaEmbedder now reuses one requests session instead of opening a new HTTP connection per call.
- Real local Ollama bge-m3, 30 warm single-query calls: P50 14.927 ms, P95 16.136 ms, maximum 16.543 ms.
- The low-cost warm-P95 performance target now passes on the measured host.
- The earlier 1703.281 ms measurement remains in this report as superseded diagnostic evidence, not silently deleted.

### Semantic category evidence

- 150 real bge-m3 requests were evaluated: 50 literature_find, 50 literature_compare, 50 general_chat.
- Top-category accuracy before release thresholds: literature_find 47/50 (94%), literature_compare 44/50 (88%), general_chat 50/50 (100%).
- Under the immutable package thresholds, all three categories produced zero auto-releases.
- Therefore the per-category 50-release and >=98% precision gate does not pass. All category switches remain disabled.

### Rebuilt package and fresh validation

- The first 150 expressions were moved to development use and the seed package was expanded to 10 find clusters, 10 compare clusters, and 15 chat clusters.
- A fresh 180-expression validation set used different topics and expressions: 60 per category.
- literature_find: 60 releases, 60 correct, precision 100%.
- literature_compare: 59 releases, 58 correct, precision 98.305%.
- general_chat: 56 releases, 55 correct, precision 98.214%.
- All categories meet the protocol's point-estimate precision and minimum-release counts; warm P95 was 18.032 ms.
- The validation set is agent-authored under user authorization, not third-party blind evidence. The user subsequently gave explicit authorization to enable all three validated categories.

### Upgrade and end-to-end latency

- Five real local qwen2.5:7b atomic-patch upgrade cases all committed successfully.
- Observed whole-command times were 4490.559, 2397.736, 2503.486, 2790.692, and 4254.421 ms.
- P50 was 2790.692 ms and P95 was 4490.559 ms, below the initial 15-second interaction target. This is a five-case smoke result, not a production distribution.
- After the WSL `es-bm25` container was started, Windows localhost:9200 and the Python Elasticsearch 8.15 client both connected successfully. The full 80-case runtime corpus then completed.
- End-to-end latency: P50 96.035 ms, P95 627.491 ms.
- Only 28/80 cases passed (35%). Integrity and citation checks passed, but functional failures remain: 47 missing grounded evidence, 36 unexpectedly waiting for clarification, 5 unsupported exits, and 5 inventory failures (reasons may overlap).

## Remaining gate evidence

- An agent-authored 200-item v2 core-routing corpus is available with exact 100/50/25/25 strata. It passed the configured core comparison, but is not represented as third-party blind evidence.
- The rebuilt package meets the configured sample-count and point-estimate precision gates on the agent-authored fresh validation set.
- Upgrade-understanding has a five-case real local smoke measurement; a larger distribution remains desirable.
- Elasticsearch connectivity is resolved. End-to-end latency passes the measured target, but the 80-case functional success gate fails and remains a V1-A blocker.
- V1-B is not admitted while these V1-A release gates remain unmet.
- Reproducible corpus, baseline, candidate, and JSON report artifacts are stored under `data/routing/v1a_acceptance` and `docs/v1a-acceptance-report-v2.json`.

## Core routing corpus v2

- Corpus digest: `8bda57e4a21244df4775ff5a911d7bfabc4f529e70c241b41243b85ca3cdd2d1`.
- Candidate: zero automatic-release errors and zero critical errors.
- Candidate: zero unnecessary clarifications and zero incorrect rejections.
- Candidate multi-turn completions: 50/50, versus baseline 31/50.
- Candidate correct automatic releases: 145, versus baseline 132.
- The reported sub-millisecond P95 measures deterministic in-process routing only. It does not supersede the real bge-m3 warm P95 of 1703.281 ms.

## Verification

- Fresh focused V1-A acceptance, routing, semantic, budget, and execution-graph selection: 31 passed.
- A wider probe produced 47 passes and 5 failures: the two existing Windows multiprocessing/request-replay failures plus three baseline/stale architecture-test incompatibilities unrelated to the new V1-A assertions.
- Python compileall, browser JavaScript syntax, patch whitespace, and the real 9x1024 matrix package load passed.
- These checks validate the implemented prototype path; they do not satisfy the frozen-corpus, per-category precision, or warm-P95 release gates.
# V1-A 业务链路修复 V1.1（2026-09-20）

本轮按照 `STUDY_V1A_业务链路修复方案_V1.md` 实施了诊断/上下文、单任务主链路和业务结果三个补丁组的核心部分。

## 已落地

- 路由语料不再把领域正确写成完整多轮任务完成；新增 `domain_correct` 和未测量标记。
- E2E 记录 request、装配 profile、AcceptedIR 摘要、源绑定、阻断码、计划、绑定/执行技能、检索状态、业务结果、分页和首断点。
- 年份完整新请求不再误判 PATCH；年份片段、上下文动作、旧结果对象和明确新话题按执行效果选择。
- 文献语义在 nlu_v2 engine 内合并并形成单一 `accepted_ir_digest`；adapter 不再二次调用原文文献解析器。
- 当前授权 Catalog 自动绑定唯一只读逻辑文献源，并记录 `current_authorized_catalog` 来源和能力。
- LogicalPlan 的文献节点均引用同一 AcceptedIR digest。
- inventory 使用独立只读动作，按固定 generation 返回总数、items、cursor、has_more、truncated 和 complete。
- RuntimeState 与 `business_outcome` 分离，已覆盖 answered/listed/no_match/document_absent/needs_clarification/unsupported 等出口。
- 修复运行时遗漏 `kb_context_effect` 导致正确上下文在编译前被清空的问题。

## 最新证据

- 相关代码回归：61 passed。
- 固定 80 例真实 ES/Chroma 检索管线：80/80 passed。
- 引用来源准确率：100%；已知页码准确率：100%；未知页码伪造：0。
- 总延迟 P50 691.419 ms，P95 923.406 ms。
- 报告：`docs/v1a-runtime-e2e-business-chain-v1.json`。

## 证据边界

- 80 例已经参与诊断，是固定回归集，不再称为独立盲测。
- 本次完整 80 例关闭外部回答模型，验证的是统一理解、计划、真实检索、证据结构和确定性合成，不等同于真实回答模型质量验收。
- 方案要求的 12 条封存表达和 6–8 条真实回答模型烟雾尚未执行，因此 V1-A 暂不冻结。
- 全 `tests` 收集仍被历史 `tests/test_enterprise_runtime_v1.py` 对已移除 `_requires_prior_comparison` 的导入阻断；本轮相关测试集独立通过。

## 最终权威验收（2026-09-21）

最终权威结论见 `docs/v1a-final-acceptance-report.json`；此前报告均作为历史诊断证据保留，不再代表当前状态。

- 全新 v4 封存表达首次且唯一一次正式运行：12/12。v2、v3 因参与过诊断，仅作为开发回归集。
- 生产 `build_application` 真实回答模型烟测：8/8，覆盖主题问答、显式文档、总结、显式 A/B 比较、多轮引用、无结果、必要澄清和库存。
- 固定 80 例真实 ES/Chroma：74/80，达到 92.5% 发布阈值；总 P50 502.471 ms，P95 716.488 ms；引用/来源和已知页码准确率 100%，伪造页码 0。
- 其余 6 例均为语料确实缺少每文档/每维度比较证据，正确返回 `insufficient_evidence`，不按错误回答计，也不放宽门槛。
- V1-A 聚焦回归：119 passed。全量回归：452 passed、23 failed、33 subtests passed；剩余失败属于 legacy Windows 句柄清理、过期路由/计划夹具、manifest/MCP 环境，不是 V1-A 发布门槛。
- 业务终止出口已统一通过中央 finalizer 记录 answered/listed/no_match/document_absent/insufficient_evidence/needs_clarification/unsupported/failed。
- V1-A 正式验收通过；V1-B 尚未启动，必须另行执行其事务可行性门槛。
# 2026-09-20 V1-A business-chain continuation

- Added SourceSpan requirement/plan linkage and enum-safe two-phase action ownership.
- Added compare object-by-dimension evidence coverage; incomplete matrices terminate as `insufficient_evidence`.
- Added engine-AcceptedIR-derived compatibility diagnostics without restoring a second parser.
- Added production `build_application` enforce-mode assembly coverage and gateway compatibility checks.
- Focused V1-A suite: 80 passed.
- Real ES/Chroma E2E: 74/80, release threshold passed, citation/source accuracy 100%, known-page accuracy 100%, total P95 1012.211 ms. Six comparison cases remain intentionally fail-closed for incomplete dimensional evidence.
- Formal freeze remains blocked on untouched holdout and 6–8 real-answer production-path smoke cases.
