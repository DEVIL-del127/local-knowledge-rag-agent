Project: kb-agent
Document-Type: independent-review
Version: 1
Status: changes-required
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: none
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Meitner (independent delegated reviewer)
Reviewed-Document: 2026-08-25_054_detail-plan_m1-1-semantic-compiler-and-patch-protocol_v1.md
Review-Round: 1
Reviewer-Identity: Meitner / process 01a03781-f616-7d50-9cb9-ff186485aeab
Reviewer-Perspective: execution boundary, compatibility regression, termination and rollback
Independence-Basis: separate delegated process; no access to peer findings
Verdict: changes-required

# Review

Scope: frozen local, read-only, single-source semantic compiler slice only.

## P1-1: Gold baseline is frozen too late

Scope-Class: current. Violated clauses: 12.2, 13 and Slice E. Creating Gold after Slice B/C allows
implementation output to redefine the expected result. Required change: freeze minimal relevant Gold
before each semantic slice, recording test hashes, denominators, exclusions, and human-pending choices.

## P1-2: IR 2.4 and terminal states lack a compatibility contract

Scope-Class: current. Violated clauses: 9 and 11. Old callers may reject `unsatisfiable/error`, and
legacy serializers may not read typed SymbolRef, OutputRef, or EventSpec. Required change: define
`ir_schema_version`, 2.4-to-legacy status/field mappings, fixture/CLI/legacy_adapter tests, and blocked
fallback when lossless downgrade is impossible.

## P1-3: rollback switch does not cover each new stage

Scope-Class: current. Violated clauses: 14 and 15. Disabling Patch cannot roll back a faulty new
TemporalNormalizer, ConstraintSolver, or OutputRef path. Required change: separate Temporal,
Constraint, Event/Set, Patch, and TurnDirective feature flags; preserve old paths through M1 exit and
test that disabled output equals the frozen legacy baseline.

Verdict: changes-required. No concurrency, permission, security-policy, or real-tool blocker was raised.
