Project: kb-agent
Document-Type: independent-review
Version: 1
Status: changes-required
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: none
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Kepler (independent delegated reviewer)
Reviewed-Document: 2026-08-25_054_detail-plan_m1-1-semantic-compiler-and-patch-protocol_v1.md
Review-Round: 1
Reviewer-Identity: Kepler / process 01a03781-f111-75b3-95e0-1b55da39c095
Reviewer-Perspective: IR contract, generic semantic correctness, testability
Independence-Basis: separate delegated process; no access to peer findings
Verdict: changes-required

# Review

Scope: local single-user, read-only, single-source semantic compiler slice only.

## P1-1: TurnDirective cannot represent compound actions

Scope-Class: current. Violated clauses: 6.5 and Slice D. “停止刚才任务，只查8月” requires
ordered `cancel_previous + new_query`, while the proposed single enum `action` loses one action.
Required change: use ordered `pre_actions[] + primary_action`, define target turn and ContextDelta
ownership, and test missing-context serialization.

## P1-2: two explicit false-valid classes remain unspecified

Scope-Class: current. Violated clauses: 2, 8.3, 8.6 and acceptance 13.3. Case 77 needs
numeric-locale/currency-conversion prerequisites; case 78 needs two-digit-year ambiguity.
Required change: add deterministic `two_digit_year`, `numeric_locale`, and
`currency_conversion_prerequisite` ambiguity/precondition outputs and Gold assertions.

## P1-3: Partial Gold cannot support claimed graph metrics

Scope-Class: current. Violated clauses: 12.2, 13.7 and 13.10. Node counts do not prove that an
Event references the correct TemporalWindow or permit precision/recall/topology computation.
Required change: add stable required nodes, key-attribute constraints, required producer-port-consumer
edges, accepted alternatives, and explicit metric denominators. Compute graph isomorphism only over
annotated subgraphs.

Verdict: changes-required. No future-scope or assurance-escalation blocker was raised.
