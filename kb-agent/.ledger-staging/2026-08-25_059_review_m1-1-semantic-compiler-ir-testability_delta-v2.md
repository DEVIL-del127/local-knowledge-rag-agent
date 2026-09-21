Project: kb-agent
Document-Type: independent-review
Version: 2
Status: changes-required
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: 2026-08-25_055_review_m1-1-semantic-compiler-ir-testability_v1.md
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Kepler (independent delegated reviewer)
Reviewed-Document: 2026-08-25_058_detail-plan_m1-1-semantic-compiler-and-patch-protocol_v2.md
Review-Round: 2 final delta
Reviewer-Identity: Kepler / process 01a03781-f111-75b3-95e0-1b55da39c095
Reviewer-Perspective: IR contract, generic semantic correctness, testability
Independence-Basis: separate delegated process; no access to peer findings
Verdict: changes-required

# Delta Review

Reviewed SHA-256: `0D87AD416B746A4069DF75BC7EDFBD6F4599DDCE78BA7E0862CD90966748A49B`.

Round-1 ambiguity/currency prerequisites and Partial Gold findings are closed.

## P1: compound directive lacks the new turn's base context

Scope-Class: current. Round-1 TurnDirective disposition is incomplete. In “停止刚才任务，只查8月”,
`context_delta.owner=new_turn` identifies the mutation target but not whether the new turn is derived from
the previous snapshot. Applying `replace temporal=August` to an empty new turn can lose the prior query
subject and constraints.

Required correction: add `base_context_ref` or `inherit_from`; distinguish derived versus fresh new turns;
forbid silently converting `replace` to `add` when the target slot is absent; emit `context_required` or a
validation error instead. Gold must independently assert inherited fields, replaced fields, and cancelled
target.

Verdict: changes-required. No other current-scope P0/P1 was found.
