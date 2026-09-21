Project: kb-agent
Document-Type: decision
Version: 1
Status: approved
Created-At: 2026-08-25 Asia/Shanghai
Parent-Overall-Plan: 2026-08-25_049_overall-plan_enterprise-knowledge-agent_v11.md
Supersedes: none
Source-Workspace: F:\A_ShiXi\Project\STUDY\kb-agent
Author: Codex

# Decision: authorize one final M1.1 Turn context correction review

The owner explicitly authorized one final correction and review after round 2 stopped. The scope is limited
to the residual P1 recorded in `2026-08-25_061_review-synthesis_m1-1-semantic-compiler_round-2-stop_v1.md`:
the new Turn must declare its base context and strict ContextDelta semantics.

Authorization permits one immutable v3 correction and one independent IR/testability delta reviewer.
It does not permit changes to other clauses, implementation code, concurrency, permissions, security policy,
ES/Skill/MCP, or root-Agent execution. Pass authorizes implementation under the reviewed plan; fail stops
without another automatic revision. This raises the review budget by exactly one correction round.
