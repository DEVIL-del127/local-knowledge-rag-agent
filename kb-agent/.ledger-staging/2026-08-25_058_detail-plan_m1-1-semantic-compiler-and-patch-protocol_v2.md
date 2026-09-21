Project: kb-agent
Document-Type: detail-plan
Version: 2
Status: under-review
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: 2026-08-25_054_detail-plan_m1-1-semantic-compiler-and-patch-protocol_v1.md
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Codex

# M1.1 通用语义编译器与 SemanticPatch 协议详细方案 v2

除以下规范性修订外，v1 的目标、冻结边界、架构、SemanticPatch、模块划分、非目标、
验收上限和停止条件保持不变。本版只解决 round 1 的五组当前范围 P1；不得把本文件解释为
扩大到并发、权限、安全策略、真实 ES/Skill/MCP 或根 Agent 执行。

## 1. Compound TurnDirective

v1 6.5 的单值 `action` 被以下契约取代：

```text
TurnDirective
  directive_id
  pre_actions[] = cancel_previous | suspend_previous
  primary_action = new_query | refine | replace_constraints | clarify
  target_turn_id: optional
  context_delta
    owner = previous_turn | new_turn
    add[]
    replace[]
    remove[]
  required_context[]
  evidence[]
```

执行顺序固定为：验证 target/required_context -> 按数组顺序解释 pre_actions -> 创建或修改
primary turn -> 将 ContextDelta 应用于显式 owner。M1.1 只输出该结构，不执行动作。

“停止刚才任务，只查8月并按地区分组”的预期是：

```text
pre_actions=[cancel_previous]
primary_action=new_query
target_turn_id=previous
context_delta.owner=new_turn
context_delta.replace=[temporal=August, grouping=region]
```

没有 previous turn 时生成 `context_required(previous_turn)`，不得 valid。新增组合动作顺序、
缺少上下文、序列化往返和 evidence 测试。

## 2. Deterministic Ambiguity And Prerequisite Analysis

ConstraintAndAmbiguityAnalyzer 增加三个当前范围分析器：

### 2.1 TwoDigitYearAnalyzer

- 识别时间/属性语境中的两位年份，不直接展开为四位年份。
- 若 QueryContext 或显式世纪策略不能唯一解析，生成 `AmbiguitySpec(kind=two_digit_year)`。
- candidates 分别保存每个原文 span 的合法解释，不允许把同文不同角色的“05年”统一映射。
- 开放 ambiguity 导致 `needs_clarification`，不得由 LLM 自行选择。

### 2.2 NumericLocaleAnalyzer

- 对含 `,` 和 `.` 的数字按显式 locale 声明解析。
- 显式 en-US/de-DE 且解析唯一时保存规范 Decimal 和原始格式。
- locale 缺失且存在多个合法解释时生成 `numeric_locale` ambiguity。
- 禁止用 binary float 作为金额的规范存储；金额值使用 decimal string + currency。

### 2.3 CurrencyConversionPrerequisiteAnalyzer

跨币种比较形成类型化 prerequisite：

```text
CurrencyConversionPrerequisite
  amount_refs[]
  source_currency
  target_currency
  rate_time_basis = order_date | query_date | explicit_date
  rate_source_ref: optional
  status = satisfied | missing | ambiguous
```

缺少币种、订单日期/适用日期、汇率输入或 rate source 时，RequestIR 可理解完整但
`binding_status=partial/unbound`，Validation 返回 `needs_clarification`，不能生成 ready plan。

测试题 77 必须断言 locale 解析正确且汇率前置条件未满足；测试题 78 必须断言两个独立
two_digit_year ambiguity。测试题 89 断言 formula ambiguity；95 断言 unsatisfiable；83/100
断言 context_required/compound directive。由这五类 Gold 共同定义“显式误放行为 0”。

## 3. Gold-First Slice Gates

v1 Slice E 不再是首次建立 Gold 的阶段。冻结顺序修改为：

```text
Slice A0: 冻结测试源哈希、旧基线和 Patch 协议 Gold
Slice A : 实现协议闭合
Slice B0: 冻结时间、歧义、前置条件和 Constraint Gold
Slice B : 实现时间与约束
Slice C0: 冻结 Event/Set/Reference Gold 与变形组
Slice C : 实现 Event/Set/Reference
Slice D0: 冻结 TurnDirective/context Gold
Slice D : 实现会话指令输出
Slice E : 汇总评估，不创建实现后期望
```

每个 Gold 文件记录：`gold_schema_version`、测试集文件 SHA-256、case_id、标注状态、标注人、
人工待确认项、指标 inclusion/exclusion 原因。实现输出不得自动写回 Gold；修正 Gold 必须新增
版本并说明原断言错误的外部证据，不得因测试失败修改期望。

Partial Gold 扩展为：

```text
expected_status_set
required_nodes[]: semantic_id, node_type, key_attribute_constraints
required_edges[]: producer_semantic_id, port, consumer_semantic_id, input_slot
allowed_alternatives[]
required_temporal_bounds[]
required_ambiguity_codes[]
required_turn_directive
forbidden_sources[]
forbidden_nodes[]
metric_inclusion[]
```

节点 precision/recall 只对 `required_nodes` 和明确 `forbidden_nodes` 计算；未标注节点不进入
分母。边覆盖和拓扑同构只对 `required_edges` 诱导出的标注子图计算。每份报告必须输出 TP、
FP、FN、分母和排除数量，禁止只输出百分比。

## 4. IR 2.4 Compatibility Contract

新增顶层 `ir_schema_version="2.4"`，与 Prompt/Patch schema_version 分开。2.4 reader 必须读取
2.3 fixture；2.3/legacy 输出通过显式 adapter 产生，不能让旧代码直接反序列化 2.4 union。

终态降级映射：

```text
2.4 valid               -> legacy valid
2.4 needs_clarification -> legacy needs_clarification
2.4 unsupported         -> legacy unsupported
2.4 unsatisfiable       -> legacy needs_clarification + diagnostic unsatisfiable
2.4 error               -> CLI/process error；不得进入 legacy RouteResult
```

字段降级规则：

- CatalogSymbolRef 可降级为旧 resolved SymbolRef。
- HypothesisSymbolRef 降级为旧 unresolved SymbolRef，并保留 hypothesis_id diagnostic。
- OutputSymbolRef 降级为 ReferenceSpec；若旧结构无法表达 producer/port，整个 legacy plan blocked。
- 2.4 Event/Set/Aggregate 若无法无损表达，legacy_adapter 返回 blocked comparison，不得截断后 valid。
- RequestIR 的 2.4 JSON 是唯一规范表示；legacy JSON 只用于兼容测试，不得回灌为 2.4 真值。

新增 2.3 fixture -> 2.4 reader、2.4 -> legacy adapter、CLI 五终态、legacy RouteResult 阻断和
序列化往返测试。

## 5. Per-Stage Rollback Contract

EngineConfig 增加独立特性开关，默认在各 Slice 验收前关闭：

```text
enable_patch_v1
enable_temporal_v2
enable_constraint_solver
enable_event_set_v2
enable_turn_directive
```

每个阶段必须保留旧实现入口到 M1 Exit Gate。开关只选择实现，不改变输入、Catalog 或旧基线
fixture。关闭所有新开关时，在已冻结旧基线字段上输出必须等价；允许新增 trace 中的
`feature_disabled`，但 Validation 状态和旧 IR 序列化不得变化。

依赖规则：

- `enable_patch_v1` 可独立开启，但只能补当前启用阶段支持的 operation。
- `enable_event_set_v2` 依赖 `enable_temporal_v2`；配置冲突在 Engine 初始化时失败。
- `enable_turn_directive` 不依赖 Planner，且 M1.1 不执行 directive。
- 新 OutputRef planner 路径随 `enable_event_set_v2` 切换，关闭时不得泄漏到旧 Planner。

测试覆盖：全关旧基线、单阶段开启、合法依赖组合、非法组合初始化失败、每个 Slice 回滚后
旧基线等价。达到 M1 Exit Gate 后再以独立决策移除旧路径，本方案不自动删除。

## 6. Revised Acceptance And Stop Conditions

v1 验收标准继续有效，并增加：

1. 组合 TurnDirective 的动作顺序、target 和 ContextDelta owner 均有 Gold edge/attribute 断言。
2. 77、78、83、89、95、100 六题不得返回 valid；95 必须为 unsatisfiable，其余按 Gold 为
   needs_clarification 或带 directive 的 blocked 状态。
3. 所有百分比报告同时输出原始 TP/FP/FN、分母和 exclusions。
4. 2.3 fixture 可读，2.4 无损可降级部分正确映射，不能无损降级时明确 blocked。
5. 所有新特性关闭后通过冻结的旧基线等价测试。

若在 Slice B0/C0/D0 无法在实现前写出稳定的最小 Gold，停止对应 Slice 并请求产品语义确认；
不得先实现再反向生成期望。若兼容 adapter 需要把不完整 2.4 节点映射为 legacy valid，停止
兼容实现并保持 blocked。其余 v1 停止与回滚条件不变。

## 7. Round 1 Findings Disposition

- Review 055 P1-1：由第 1 节解决。
- Review 055 P1-2：由第 2 节和第 6 节解决。
- Review 055 P1-3：由第 3 节解决。
- Review 056 P1-1：由第 3 节的 A0/B0/C0/D0 gate 解决。
- Review 056 P1-2：由第 4 节解决。
- Review 056 P1-3：由第 5 节解决。

本 v2 只接受一次 delta review。Reviewer 只检查上述 disposition 是否闭合，不重新审查已在
v1 冻结且未改变的范围。若仍有当前范围 P0/P1，记录 gate-fail 并停止自动修订，交由主人决定。
