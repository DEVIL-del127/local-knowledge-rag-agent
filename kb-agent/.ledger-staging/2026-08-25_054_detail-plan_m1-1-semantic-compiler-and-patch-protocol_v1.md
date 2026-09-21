Project: kb-agent
Document-Type: detail-plan
Version: 1
Status: proposed
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: none
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Codex

# M1.1 通用语义编译器与 SemanticPatch 协议详细方案 v1

## 1. 方案定位

本方案是 M1 测试后的修正增量，不改变总体产品目标，也不提前进入 M2 Catalog
接入或 M3 根 Agent 执行。它用于闭合当前 M1 Exit Gate 中尚未达到的通用语义能力：

- LLM 输出协议与本地 Merge 的一一对应。
- 精确时间范围、连续/累计事件、集合运算和引用链。
- 公式歧义、逻辑不可满足和上下文前置条件。
- 问题内 Schema 的正确理解，以及与 Catalog 可执行绑定的分离。
- 为根 Agent 会话状态机输出稳定的 TurnDirective，但本切片不执行状态转换。

本方案吸收 LangGraph 的类型化 Partial State、Microsoft Agent Framework 的显式工作流、
OpenAI Agents SDK/Pydantic AI 的严格结构化输出，以及 Rasa CALM 的“LLM 理解、代码执行
业务逻辑”模式。不会直接引入这些框架作为运行依赖。

## 2. 当前证据

100 题规则链与 Ollama 完整链测试结果：

- 规则链和完整链状态均为 `valid=5 / needs_clarification=87 / unsupported=8`。
- 89 次 LLM 调用只有 2 题出现结构节点数量增加，最终状态无改善。
- 5 个 `valid` 均存在明确误放行：数字/币种前置条件、两位年份歧义、会话上下文、
  公式括号歧义、逻辑不可满足。
- 至少 29 题存在连续/持续语义但缺少完整 Event。
- 至少 36 题存在集合语义但缺少 SetOperation。
- “2026年1月至2026年6月”被拆为两个独立月份，Event 只继承第一个月份。
- 默认 Catalog 字段别名可诱发虚假的多数据源冲突。

协议静态核查发现：

- LLM Schema 声明 `hypothesis_references`、`sampling`、`aggregates`、`comparisons`，
  Merge 没有对应处理器。
- Merge 处理 `metrics`，LLM Schema 却没有声明 `metrics`。
- Ollama 当前只使用 `format=json`，未传入真实 JSON Schema。
- 任何可解析 JSON 都被记为调用成功并缓存，即使全部节点被拒绝或没有增益。
- Existing IR 摘要丢失 Event 条件、时长、窗口、输出引用和完整证据，模型无法可靠补链。

## 3. 根因分类

### 3.1 架构缺口

当前模型输出是“任意 IR 片段”，Merge 是字段级启发式拼接。两者之间没有可版本化、
可穷举、可测试的 Patch 协议，也没有事务语义。

### 3.2 通用能力缺失

Temporal、Event、Set、Reference、Formula 和 Constraint 分散在一个较大的规则提取器中，
缺少先后阶段与稳定 producer/output 引用。

### 3.3 验证缺口

Validator 主要验证已有节点能否绑定，没有先验证原文中的歧义、上下文依赖、逻辑可满足性
和需求结构是否完整。

### 3.4 Catalog/数据契约缺口

问题内字段已经能形成 SchemaHypothesis，但后续 Event 和 Aggregate 仍偏向要求 CatalogSymbol，
导致“理解正确但不可执行”无法表达完整。

## 4. 冻结实施边界

### 4.1 当前切片

输入自然语言和可选只读 Catalog，输出 RequestIR、ValidationReport、LogicalPlan shell、
TurnDirective 和完整 Trace。LogicalPlan 仍不执行，PhysicalPlan 仍为 unbound。

### 4.2 非目标

- 不处理 Ollama 并发、吞吐、P95/P99 和全局限流。
- 不实现企业权限、租户隔离、DLP、审批或安全策略。
- 不接 DeepSeek API、ES 检索、Skill、MCP 或真实工具。
- 不支持真实跨数据源 join；多表问题允许理解完整，但执行继续 blocked/unsupported。
- 不实现根 Agent 的持久化 RunController，只定义它未来需要消费的 TurnDirective。
- 不为 100 题逐题编写专用规则。

### 4.3 数据与副作用

仅处理测试文本、本地静态 Catalog 和本地 Ollama 结构候选。无真实企业数据、无网络副作用、
无写数据操作。风险等级为 medium，原因是可逆 IR/解析协议变更和有界本地模型调用。

## 5. 目标架构

```text
QueryEnvelope
 -> DiscourseSegmenter
 -> RequirementExtractor (仅从原文提取义务)
 -> QuerySchemaExtractor
 -> TemporalNormalizer
 -> ClauseParser
 -> Predicate/Event/Aggregate/Set Builders
 -> ReferenceResolver
 -> ConstraintAndAmbiguityAnalyzer
 -> Preliminary RequestIR
 -> GapAnalyzer
 -> optional Ollama SemanticPatchV1
 -> PatchSchemaValidator
 -> PatchEvidenceValidator
 -> TransactionalPatchApplier
 -> ClaimReconciler
 -> RequirementCoverageEvaluator
 -> IRValidator
 -> LogicalPlanner
 -> TurnDirective + unbound PhysicalPlan
```

规则层不再负责“一次生成最终结果”，而是生成类型化原子节点。后续 Builder 负责组合，
Analyzer 负责判定矛盾和歧义，LLM 只补 Gap。

## 6. 核心数据契约

### 6.1 SymbolRef 拆分

把当前依赖 `ref_kind` 字符串的 SymbolRef 收紧为可区分联合类型：

```text
CatalogSymbolRef
  catalog_id
  source_id
  binding_status
  evidence[]

HypothesisSymbolRef
  hypothesis_id
  normalized_name
  declared_type
  executable=false
  evidence[]

OutputSymbolRef
  producer_id
  port
  result_type
  shape
  grain
```

兼容层继续输出旧 `SymbolRef` 字段，直到 `legacy_adapter` 和 CLI 完成迁移。

### 6.2 TemporalWindow

TemporalItem 增加稳定 `temporal_id`。范围只存一次，Event、Aggregate 和 Calculation
通过 `window_ref` 引用，禁止复制 `temporal[0]`：

```text
TemporalWindow
  temporal_id
  operator
  lower / upper
  lower_inclusive / upper_inclusive
  granularity
  timezone
  resolution_status
  evidence[]
```

“2026年1月至6月”必须组合为一个闭区间；“截至今日”保留 runtime anchor，不在解析器中
写死日期；两个独立区间必须保持两个 TemporalWindow，不允许误合并。

### 6.3 EventSpec 2.4

```text
EventSpec
  event_id
  event_type = consecutive | cumulative | point | window
  condition: PredicateNode
  duration_constraint
  accumulation_window_ref
  temporal_window_ref
  sampling_policy_ref
  partition_by[]
  group_by[]
  output_ref
  evidence[]
```

Event 条件允许引用 CatalogSymbol 或 HypothesisSymbol。Hypothesis 事件可以语义完整，
但 `binding_status=unbound`，因此不能生成 ready 执行计划。

### 6.4 关系代数与引用

统一用 OutputRef 连接节点：

```text
Project(event_intervals -> local_date)
Intersection(left_dates, right_dates)
Union(left, right)
Difference(left, right)
Overlap(left_intervals, right_intervals)
Aggregate(input_relation, expression, group_by)
Compare(left_expression, operator, right_expression)
```

SetOperation 不再保存脆弱的字符串 `inputs` 作为唯一真值；旧字段仅用于序列化兼容。
ReferenceResolver 在 LogicalPlanner 前建立 DAG，检测缺失 producer、端口类型不匹配和依赖环。

### 6.5 TurnDirective

本切片只识别并输出，不执行：

```text
TurnDirective
  action = new_query | refine | replace_constraints | cancel_previous | clarify
  target_turn_id: optional
  context_delta
    add[]
    replace[]
    remove[]
  required_context[]
  evidence[]
```

“那英文的呢”产生 refine + language replace；没有上轮上下文时返回 context_required。
“停止刚才任务，只查8月”产生 cancel_previous + new query context delta。

## 7. SemanticPatchV1

### 7.1 GapRequest

GapAnalyzer 只从确定性 IR 和原文生成缺口：

```text
GapRequest
  gap_id
  gap_type
  target_node_id
  required_slots[]
  query_slice
  allowed_catalog_symbols[]
  allowed_hypothesis_symbols[]
  allowed_metrics[]
  available_output_refs[]
  existing_node_summary
```

首批 Gap 类型：`missing_event`、`missing_set_inputs`、`missing_formula_input`、
`missing_aggregate_scope`、`unresolved_reference`、`ambiguous_formula`、
`context_required`。未知类型不能发送给模型。

### 7.2 Patch 结构

```text
SemanticPatchV1
  schema_version
  base_ir_digest
  operations[]

PatchOperation
  operation_id
  gap_id
  operation_type
  target_node_id
  preconditions[]
  dependencies[]
  evidence[]
  payload
```

允许的 operation_type：

- `add_event`
- `add_set_operation`
- `add_aggregate`
- `add_comparison`
- `add_calculation`
- `resolve_reference`
- `add_ambiguity`
- `add_turn_directive`

不允许模型通过 Patch 创建已授权 Catalog 对象、PhysicalPlan、ToolCall 或任意 Python/SQL
表达式。计算必须使用 ExpressionNode AST。

### 7.3 严格输出

使用 Pydantic v2 定义 discriminated union，并把 `model_json_schema()` 直接作为 Ollama
`/api/chat` 的 `format`。返回后使用 `model_validate_json()`，不再把“可解析 dict”视为合法。

Prompt 只包含：当前 Gap、相关原文切片、允许枚举、相关符号、相关 existing nodes 和对应
JSON Schema。禁止发送最多 200 个无关 Catalog 字段让 3B 模型重新分析整题。

### 7.4 Patch 校验与事务合并

每个 Patch 依次通过：

1. Schema validation。
2. `base_ir_digest` 和 Gap 存在性验证。
3. EvidenceSpan 必须逐字匹配原文，允许重复文本时由显式 start/end 消歧。
4. Symbol 和 DerivedMetric 必须在允许集合内。
5. OutputRef、依赖和类型/单位一致性验证。
6. Operation 前置条件验证，禁止覆盖已接受节点。
7. 在 RequestIR 副本中应用全部操作。
8. 对副本运行 Constraint、Reference、Requirement 和 IRValidator。
9. 只有无新增致命错误且至少关闭一个目标 Gap 时原子提交。

PatchReport 记录每个 operation 的 accepted/rejected、原因、关闭的 gap 和新增节点。只有
`accepted_count > 0 && closed_gap_count > 0` 才算 LLM 有效增益、成功调用和可缓存结果。

默认最多一次主调用。可选 Schema 修复只修 JSON/类型错误，不重新理解语义，总调用上限仍为 2。

## 8. 确定性语义组件

### 8.1 DiscourseSegmenter

把输入切成 instruction、constraint、quoted_content、example、expected_behavior、schema_declaration。
这里只解决语义归属，不实现安全策略。否定范围和引用内容不进入错误的操作意图。

### 8.2 QuerySchemaExtractor

提取反引号字段、字段中文说明、枚举、采样声明和表结构。连接词“以及、且、累计、连续、
总时长”不能成为字段前缀。问题声明字段优先形成 HypothesisSymbol，但不覆盖 Catalog。

### 8.3 TemporalNormalizer

先识别完整区间，再识别单点；执行 longest-span-first 和 overlap suppression。支持年月范围、
精确日期、相对时间、两个独立区间、时区和运行时 anchor。所有消费者只引用 temporal_id。

### 8.4 EventBuilder

以 Clause 为单位匹配条件和持续/累计约束，不以全句最近字段猜测。支持：

- `CONSECUTIVE(condition, duration)`
- `CUMULATIVE_DURATION(condition, calendar_window, duration)`
- 同期两个条件的复合 Event
- 采样间隔、缺失中断、分区键和日期投影

同一模板替换成温度、压力、速度、功率、库存或加速度后必须产生同构 IR。

### 8.5 FormulaAndAggregateBuilder

使用 ExpressionNode 表达 scoped aggregate、算术和比较，不使用自由字符串。命名公式注册表
只保存语义定义，例如 stddev、ratio、growth、drawdown。原文公式存在多种 AST 时产生
AmbiguitySpec，不自动选择执行。

### 8.6 ConstraintSolver

第一阶段只做可判定的局部约束：同字段上下界、eq/ne 冲突、空集合、类型/单位不兼容、
必需上下文缺失和明显公式歧义。结果为 satisfiable、unsatisfiable、ambiguous 或 unknown。
unknown 不能伪装成 valid。

## 9. Engine 新流程与终止条件

Engine 每个查询只允许以下有向流程：

```text
deterministic compile
 -> gap analysis
 -> zero or one semantic model call
 -> optional one schema-only repair
 -> transactional validation
 -> final validation
 -> terminal result
```

禁止模型输出触发新的模型调用，禁止 Validator 自动重新规划，禁止同一 Gap 重复补全。
终态为 `valid`、`needs_clarification`、`unsupported`、`unsatisfiable` 或 `error`。

## 10. 预计文件改动

### 修改

- `nlu_v2/models.py`：Schema 2.4、typed symbol、TemporalWindow、EventSpec、TurnDirective、
  GapRequest、SemanticPatch、PatchReport。
- `nlu_v2/engine.py`：新阶段编排、Gap 调用、事务提交、有效增益缓存。
- `nlu_v2/llm_extractor.py`：Pydantic JSON Schema、Gap prompt、严格返回校验。
- `nlu_v2/merge.py`：替换为 PatchOperation 注册表和事务 PatchApplier；保留旧入口适配期。
- `nlu_v2/rule_extractor.py`：收缩为原子 Clause/Predicate/Schema 提取，复用成熟规则。
- `nlu_v2/claims.py`：对新节点生成稳定 claim_id，避免同源重复。
- `nlu_v2/requirements.py`：加入上下文、歧义、集合输出和公式义务映射。
- `nlu_v2/validator.py`：消费 Constraint、Reference 和 PatchReport，扩展终态。
- `nlu_v2/logical_planner.py`：按 OutputRef DAG 规划并独立 preflight。
- `nlu_v2_validate.py`：展示 Gap、PatchReport、Constraint 状态和 TurnDirective。

### 新增

- `nlu_v2/discourse.py`
- `nlu_v2/query_schema.py`
- `nlu_v2/temporal_normalizer.py`
- `nlu_v2/event_builder.py`
- `nlu_v2/expression_builder.py`
- `nlu_v2/reference_resolver.py`
- `nlu_v2/constraint_solver.py`
- `nlu_v2/gaps.py`
- `nlu_v2/patch_protocol.py`

模块应保持单一职责；不再继续扩张 `rule_extractor.py` 或新建另一个巨型文件。

## 11. 兼容与迁移

- `RequestIR = UnderstandingIR` 兼容别名暂时保留。
- 旧 `merge_llm_candidate()` 在一个过渡版本中转换旧 payload 为 LegacyPatch；默认新 Engine
  不再调用旧自由片段协议。
- CLI 默认行为仍为完整链，`--no-llm` 保持可用。
- Legacy adapter 继续只接受简单、valid、单源 private_kb 计划。
- 旧测试先保持通过，再逐步增加 2.4 断言；不得通过删除旧断言制造通过。
- 如果新 Patch 链出现回归，可通过 EngineConfig 临时切回 deterministic-only，不回退 IR 数据。

## 12. 测试设计

### 12.1 单元测试

- Pydantic Schema 与每种 PatchOperation 正反例。
- Prompt 声明的每个 operation 都有对应 applier；applier 注册表和 union 枚举集合必须相等。
- Temporal longest-span、双区间、精确终点和 runtime anchor。
- Event 条件/时长/采样/窗口组合。
- Set/Project/Reference DAG、缺失 producer、类型错误和循环。
- Constraint satisfiable/unsatisfiable/ambiguous/unknown。
- QuerySchema 字段清洗和 Hypothesis-only 完整理解。
- Patch 全收、部分拒绝、全部拒绝、原子回滚、无增益不缓存。

### 12.2 100 题 Partial Gold

不要求一次人工标注完整 IR，先为每题记录：

```text
expected_status_set
required_node_types
required_temporal_bounds
required_event_count
required_set_operation
required_ambiguity_codes
required_turn_directive
forbidden_sources
forbidden_nodes
```

自动报告规则链、LLM 链、Patch 合法率、Patch 有效增益率、节点 precision/recall、误放行、
误阻断、结构同构率。没有 Gold 的字段不计算准确率。

### 12.3 变形测试

- 加速度替换压力、速度、功率、库存和温度。
- AND/OR、语序、单位、中文/英文别名变化。
- 同一 Event 使用 CatalogSymbol 与 HypothesisSymbol。
- 引用词替换为“这些日期、上述结果、前一步结果”。
- 在问题中加入示例、否定约束和引用内容，核心 IR 不被污染。

## 13. 验收标准

必须同时满足：

1. 当前全部自动化测试通过，无源代码异常。
2. Prompt/Patch union/Applier operation 集合完全一致，不存在“模型能输出但 Merge 忽略”的字段。
3. 100 题中明确矛盾、歧义和上下文依赖的误放行数为 0。
4. Gold 中精确日期和时间范围边界匹配率为 100%。
5. 有明确完整条件和时长的连续/累计问题，Event 结构覆盖率不低于 90%。
6. 有两个可识别输入的交集/并集/差集问题，SetOperation 结构覆盖率不低于 90%。
7. 所有 OutputRef 均可解析或产生明确 blocked diagnostic；不得悬空进入 ready plan。
8. 成功返回的 Ollama 结果中，严格 SemanticPatch 合法率不低于 90%。
9. LLM 不得覆盖规则确认的精确时间边界，不得使规则链已有正确节点退化。
10. 至少五个领域替换组的 IR 拓扑同构率不低于 95%。
11. 单题模型调用默认不超过 1，开启 schema repair 后总调用不超过 2。
12. 完整链不能因 LLM 失败而抛出未处理异常。

LLM 有效增益率作为观测指标，不在第一轮设置虚假高门槛；若合法率达标但有效增益仍低于
20%，应停止扩大 Prompt，进入模型能力/任务拆分评估，不通过添加领域特化规则掩盖问题。

## 14. 实施切片

### Slice A：协议闭合

实现 GapRequest、SemanticPatch、严格 Schema、Operation 注册表、PatchReport 和事务提交。
先解决“协议不一致”，暂不重写语义解析。

### Slice B：时间与约束

实现 TemporalNormalizer、ConstraintSolver、公式歧义和 context_required，消除 5 个错误 valid。

### Slice C：Event/Set/Reference

实现通用 EventBuilder、日期投影、集合 DAG 和引用解析，修复时序及集合大面积漏识别。

### Slice D：会话指令输出

实现 DiscourseSegmenter 和 TurnDirective，只做识别与验证，不启动根 Agent 状态机。

### Slice E：100 题 Gold 验收

建立 Partial Gold、变形测试和规则/LLM 双轨报告。达到 M1 Exit Gate 后才讨论 M2 Catalog。

每个 Slice 必须可独立回归和回退，禁止五个 Slice 一次性大爆炸合并。

## 15. 停止与回滚条件

出现以下任一情况，停止当前 Slice 并报告，不自动扩大范围：

- 为通过样例需要引入领域字段硬编码。
- 新 Schema 无法兼容旧 RequestIR 且没有适配器。
- Patch operation 无法通过统一类型和证据验证，只能恢复任意 dict Merge。
- 新流程使误放行增加，或规则链正确时间边界被 LLM 覆盖。
- 单一 Slice 超过其目标并开始实现真实 Catalog、Skill、MCP 或根 Agent 执行。
- 连续两轮修订没有减少当前范围 P0/P1。

回滚以 EngineConfig 关闭新 Patch/Builder 为主，不删除旧文件、不回退用户改动、不执行
破坏性 Git 操作。

## 16. 需要主人提供或设置的内容

当前设计和 Slice A-C 无需主人新增配置，继续使用已部署的：

```text
Ollama URL: http://127.0.0.1:11434
NLU model: qwen2.5:3b
Python: 项目现有 WSL venv
```

实施前会自动检查项目环境是否已安装 `pydantic>=2`。项目根 `pyproject.toml` 已声明
`pydantic>=2.13.4`；只有 WSL venv 尚未同步依赖时，才需要主人执行或允许执行 `uv sync`
或对应 pip 安装。

主人暂时不需要准备 ES mapping、企业权限、MCP 或真实业务数据。进入 Slice E 时，需要主人
确认 100 题中少数产品语义选择，例如两位年份默认策略、公式歧义是否允许推荐默认值；在确认
之前这些样例按 needs_clarification 验收，不阻塞基础实现。

## 17. 审查与授权

本方案属于 medium 风险的架构修正。若进入实施，按当前项目规范对本 exact v1 做一次有界审查：

- Reviewer 1：IR 契约、语义完整性与可测试性。
- Reviewer 2：执行边界、回归风险与停止条件。

每位最多 3 个当前范围 P0/P1；P2 和未来 M2-M5 不阻塞。若审查要求扩大到并发、生产权限、
多租户或真实工具执行，应归类为 deferred，不得扩张本方案。双 pass 后按 Slice A 开始代码。
