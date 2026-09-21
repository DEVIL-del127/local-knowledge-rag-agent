Project: kb-agent
Document-Type: detail-plan
Version: 3
Status: under-review
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: 2026-08-25_058_detail-plan_m1-1-semantic-compiler-and-patch-protocol_v2.md
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Codex

# M1.1 通用语义编译器与 SemanticPatch 协议详细方案 v3

v2 的全部已通过条款保持不变。本版只修复 round 2 唯一剩余 P1，不扩大范围。

## 1. Normative TurnDirective Contract

v2 §1 的 TurnDirective 被以下完整契约取代：

```text
TurnDirective
  directive_id
  pre_actions[] = cancel_previous | suspend_previous
  primary_action = new_query | refine | replace_constraints | clarify
  target_turn_id: optional
  context_mode = fresh | inherit
  base_context_ref: optional TurnContextSnapshotRef
  context_delta
    owner = previous_turn | new_turn
    add[]
    replace[]
    remove[]
  required_context[]
  evidence[]
```

Normative invariants:

1. `context_mode=inherit` requires a resolved `base_context_ref`; absence produces
   `context_required(base_turn_snapshot)` and cannot be valid.
2. `context_mode=fresh` forbids `base_context_ref`. A fresh turn may use `add`, but cannot use `replace`
   for a slot that has no value in its working context.
3. `base_context_ref` points to an immutable semantic `TurnContextSnapshot`, not the mutable execution
   object and not late tool output.
4. ContextDelta belongs to its explicit owner. A delta for `new_turn` cannot mutate the previous turn.
5. `replace(slot, value)` is strict: the slot must exist in the inherited or current working context.
   Missing targets produce `context_delta_target_missing`; replace never silently degrades to add.
6. `add(slot, value)` fails with `context_delta_target_exists` when the slot is single-valued and already
   exists; callers must use replace. Multi-valued slots follow their declared merge policy.
7. `remove(slot)` requires the slot to exist, otherwise produces `context_delta_target_missing`.

## 2. Normative Interpretation Order

NLU V2 only produces and validates this directive; the future root Agent consumes it in this fixed order:

```text
resolve target_turn_id and required_context
 -> resolve and freeze immutable base_context_ref snapshot
 -> validate all delta operations against a copy of the base/fresh context
 -> interpret pre_actions in listed order
 -> instantiate primary turn from validated working context
 -> emit directive result
```

Validation happens before cancellation interpretation, so a malformed follow-up cannot cancel the previous
turn and then fail because its context was unavailable. M1.1 does not execute these transitions; the order is
part of the output contract for future M3.

## 3. Required Example

Given previous semantic context:

```text
turn_id=turn_41
subject=sales
metric=sales_total
temporal=year_to_date
grouping=[]
```

and user input “停止刚才任务，只查8月并按地区分组”, the required directive is:

```text
pre_actions=[cancel_previous]
primary_action=new_query
target_turn_id=turn_41
context_mode=inherit
base_context_ref=turn_41.semantic_snapshot
context_delta.owner=new_turn
context_delta.replace=[temporal=August]
context_delta.add=[grouping=region]
```

The resulting validated working context retains `subject=sales` and `metric=sales_total`, replaces only
temporal, and adds grouping. It must not retain `year_to_date`, must not mutate turn_41, and must not bind
late results from turn_41.

If no prior snapshot exists, output `context_required(base_turn_snapshot)` with no executable directive.
If the inherited context has no temporal slot, `replace temporal=August` produces
`context_delta_target_missing`; it does not become add. The parser may propose an explicit add candidate only
as a separate ambiguity/clarification outcome, not silently rewrite the operation.

## 4. Gold And Tests

Slice D0 Gold must separately assert:

- `target_turn_id` and `base_context_ref` identify the same previous semantic snapshot.
- inherited slots retained: subject and metric.
- replaced slots changed and old values removed: temporal.
- added slots introduced: grouping.
- previous-turn context remains immutable.
- absent snapshot produces context_required.
- strict replace/add/remove missing/existing target diagnostics.
- serialization round trip preserves context_mode, base_context_ref, delta owner and operation order.

These assertions join v2 `required_nodes`, `required_edges` and key-attribute constraints; they are frozen in
Slice D0 before implementation. No other v2 Gold or acceptance denominator changes.

## 5. Disposition And Stop

This closes review 059 P1 by defining context origin, immutable snapshot semantics, strict delta operations,
interpretation order and explicit Gold. One independent IR/testability reviewer checks only these clauses
against the residual P1. A pass is the exact-version gate-pass for v3 together with all v2 clauses inherited
unchanged. A fail stops M1.1 review; no v4 is created automatically.
