# Requirements Rubric — Completeness Checklist

Every item here maps to a specific spec requirement. Check against this before every major milestone.
Status: [ ] = not started, [~] = partial, [x] = done

---

## A. SYSTEM ARCHITECTURE

- [ ] A1. Three agents: Assessment (chat), Resolution (voice), Final Notice (chat)
- [ ] A2. Temporal workflow orchestrating 1 workflow per borrower
- [ ] A3. Linear pipeline: Assessment → Resolution → Final Notice
- [ ] A4. Outcome-based transitions per flowchart (deal_agreed → exit, no_deal → Final Notice, etc.)
- [ ] A5. Retry logic: max 3 attempts for Assessment on no response
- [ ] A5a. If Assessment no-response retries are exhausted, workflow still proceeds to Resolution (per flowchart)
- [ ] A5b. Temporal timeouts defined per stage and handled explicitly
- [ ] A6. Borrower never feels a handoff — no re-introductions, no repeated questions, no tone shifts
- [ ] A7. Agent personality fidelity: Agent 1 cold/clinical, Agent 2 transactional, Agent 3 consequence-driven
- [ ] A8. Temporal persists borrower state, transcripts, summaries, and outcome across stages

## B. CONTEXT BUDGET (hard constraint, enforced in code)

- [ ] B1. 2000 tokens total per agent (system prompt + handoff context)
- [ ] B2. Max 500 tokens for handoff summary
- [ ] B3. Agent 1: full 2000 tokens (no prior context)
- [ ] B4. Agent 2: 500 handoff + 1500 system prompt
- [ ] B5. Agent 3: 500 handoff (covering Agent 1 + Agent 2 history) + 1500 system prompt
- [ ] B6. Enforced in code with tiktoken — not aspirational
- [ ] B7. Summarizer preserves: identity verified, financial situation, offers made, objections raised, emotional state
- [ ] B8. If summary drops something important → next agent asks again → we detect and penalize this
- [ ] B9. Budget enforcement is evidenceable: logs/artifacts show prompt tokens, handoff tokens, and hard-fail behavior
- [ ] B10. Agent 3 handoff budget is enforced on the combined Agent 1 + Agent 2 history, not 500 + 500

## C. CROSS-MODAL HANDOFF

- [ ] C1. Agent 1 → Agent 2: voice call picks up exactly where chat left off
- [ ] C2. Agent 2 already knows who borrower is, what they owe, their situation
- [ ] C3. No re-verification in Agent 2
- [ ] C4. Agent 2 → Agent 3: everything from phone call captured and passed
- [ ] C5. Agent 3 references what was discussed on the call
- [ ] C6. Chat thread reads as coherent continuation
- [ ] C7. Architectural decision for handoff mechanism documented and justified
- [ ] C8. Voice call transcript/log is captured reliably enough for downstream summarization and audit
- [ ] C9. Agent 2 → Agent 3 handoff includes offers, deadlines, objections, borrower stated position, and any hardship/distress signals
- [ ] C10. Next agent uses handoff context instead of asking borrower to repeat known facts

## D. SELF-LEARNING LOOP (4 hard requirements)

- [ ] D1. **Quantitative justification** — numbers, not "LLM said it's better"
- [ ] D2. **Statistical rigor** — "40% to 45%" not sufficient without significance testing
- [ ] D3. **Compliance preservation** — no prompt update may introduce violations
- [ ] D4. **Audit trail** — every prompt version stored with its evaluation data
- [ ] D5. **Rollback capability** — revert to previous version if underperformance
- [ ] D6. **System-level evaluation** — not just per-agent; changes to one agent can degrade handoff/continuity
- [ ] D7. Prompt change impact on handoff experience explicitly evaluated
- [ ] D8. Test harness covers: cooperative, combative, evasive, confused, distressed borrowers
- [ ] D9. Prompt update is adopted only if improvement holds across all borrower scenarios; no scenario/persona type regresses

## E. DARWIN GODEL MACHINE (meta-evaluation)

- [ ] E1. System evaluates AND improves its own evaluation methodology
- [ ] E2. Metrics might be wrong → system can identify this
- [ ] E3. Threshold for adopting changes might be wrong → system can identify this
- [ ] E4. Compliance checker might have blind spots → system can identify this
- [ ] E5. **At least 1 concrete demo**: meta-eval caught a flaw in primary evaluation and corrected it
- [ ] E6. Show what was misleading/lenient/blind, and what the correction was
- [ ] E7. Meta-eval checks for evaluator gaming / objective hacking, not just low-quality metrics
- [ ] E8. Meta-eval changes are themselves versioned, justified, and auditable

## F. COMPLIANCE (8 rules, all agents, including after prompt updates)

- [ ] F1. Identity disclosure — AI identification at start of conversation
- [ ] F2. No false threats — never threaten legal/arrest/garnishment unless documented next step
- [ ] F3. No harassment — acknowledge and flag if borrower asks to stop contact
- [ ] F4. No misleading terms — settlement offers within policy-defined ranges
- [ ] F5. Sensitive situations — hardship/medical/crisis → offer hardship program, no pressure
- [ ] F6. Recording disclosure — inform conversation is being logged/recorded
- [ ] F7. Professional composure — maintain professional language regardless of borrower behavior
- [ ] F8. Data privacy — never display full account numbers, SSN, etc.
- [ ] F9. Compliance preserved AFTER every prompt update (re-check after evolution)
- [ ] F10. Compliance as hard gate — any violation = variant rejected
- [ ] F11. Consequences stated by Agent 3 are authorized by documented workflow/account state, not static prompt text
- [ ] F12. Stop-contact / refusal flags propagate across later retries and stages, not just the current turn
- [ ] F13. Hardship/distress handling includes stopping pressure, not just mentioning hardship referral

## G. RUBRIC & EVALUATION QUALITY

- [ ] G1. Rubrics cover all scoring dimensions: goal completion, compliance, quality, handoff, system continuity
- [ ] G2. Rubrics are VARIABLE — but ONLY by the meta-eval cycle (Phase 15), NEVER by the evolution loop (Phase 11-12). Prevents optimizer from gaming its own evaluation.
- [ ] G3. Ceiling/floor detection — if a metric is always 10 or always 1, rubric needs refinement
- [ ] G4. Cross-scenario robustness — agents must score well across ALL persona types, not just one
- [ ] G5. Aggregate trajectory analysis — rewriter sees patterns across all conversations, not just worst cases
- [ ] G6. Two-tier evaluation prevents gaming — cheap scorer + independent strict grader
- [ ] G7. Rubric mutation has guardrails — can't weaken compliance, must pass anchor tests
- [ ] G8. Per-conversation scores available, not just aggregates
- [ ] G9. Distribution of outcomes shown, not just means — variance matters ("8% improvement with 40% std dev is not an improvement")
- [ ] G10. Scoring weights are themselves variable and evolvable
- [ ] G11. Rubrics explicitly check continuity seams: re-introductions, re-verification, repeated questions, tone discontinuity
- [ ] G12. Voice-specific evaluation exists for Agent 2 (call opening, objection handling, spoken continuity, transcript fidelity)
- [ ] G13. Fair comparison rule: rubrics are FIXED during evolution, so parent and child always share the same eval config. After meta-eval changes rubrics, entire archive is re-scored.
- [ ] G14. Regular scorer and strict grader disagreement is logged and reviewed by meta-eval

## H. DELIVERABLES

### H1. Working System
- [ ] H1a. Temporal workflow orchestrating 3-agent pipeline
- [ ] H1b. Two chat agents and one voice agent (functional)
- [ ] H1c. Cross-modal handoff with context summarization
- [ ] H1d. Test harness for generating and evaluating conversations
- [ ] H1e. Self-learning loop with meta-evaluation capability
- [ ] H1f. Docker Compose: full system runs on fresh machine in <5 minutes
- [ ] H1g. System live and runnable at all times after submission
- [ ] H1h. Can trigger conversations through the system for live evaluation
- [ ] H1i. Borrower-facing experience does not expose internal handoff summaries or other internal seams

### H2. Evolution Report (per agent)
- [ ] H2a. How prompt evolved over iterations
- [ ] H2b. Quantitative data behind each adoption or rejection
- [ ] H2c. Metrics across prompt versions
- [ ] H2d. Regressions detected and how handled
- [ ] H2e. At least 1 meta-eval catch demonstrated
- [ ] H2f. Total LLM API spend with cost breakdown
- [ ] H2g. Raw numbers, not summaries
- [ ] H2h. Per-conversation scores
- [ ] H2i. Distribution of outcomes + variance
- [ ] H2j. Treat like scientific experiment

### H3. Reproducibility
- [ ] H3a. Exact seed/config to regenerate test conversations
- [ ] H3b. Single command reruns evaluation pipeline end-to-end
- [ ] H3c. Raw data files (CSV/JSON) with per-conversation scores
- [ ] H3d. Reported numbers match within reasonable tolerance on rerun
- [ ] H3e. Pinned model versions (not just "gpt-4o-mini")

### H4. Technical Writeup
- [ ] H4a. Architecture: system design, cross-modal handoff, context flow under 500-token budget
- [ ] H4b. Self-learning approach: what measured, how evaluated, why designed this way
- [ ] H4c. Meta-evaluation: how loop evaluates itself, what it caught, what it changed
- [ ] H4d. Compliance: how prompt updates don't violate rules
- [ ] H4e. Limitations: what doesn't work, what you'd improve

### H5. Other Deliverables
- [ ] H5a. Public GitHub repo with detailed README
- [ ] H5b. Decision journal (handwritten, timestamped)
  - [ ] 3+ architectural decisions with alternatives tried
  - [ ] 2+ moments of being wrong/stuck with recovery
  - [ ] 1+ intentional non-build decision with rationale
- [ ] H5c. Audio recording of conversation with voice Agent 2
- [ ] H5d. Short demo (2-3 minutes) walking through system
- [ ] H5e. Cost breakdown of LLM API spend

## I. CONSTRAINTS

- [ ] I1. $20 total LLM API spend for entire learning loop
- [ ] I2. Report actual cost with breakdown: conversations simulated, eval calls, prompt-gen calls
- [ ] I3. 5-day deadline
- [ ] I4. TypeScript or Python (or both)
- [ ] I5. Temporal required
- [ ] I6. No starter kit — built from scratch
- [ ] I7. Docker Compose required
- [ ] I8. Cost estimates are backed by token/accounting logs, not rough per-conversation guesses

## J. INTERVIEW PREPAREDNESS (non-code but critical)

- [ ] J1. Can navigate own codebase fluently without AI
- [ ] J2. Can explain statistical methodology (bootstrap, Wilcoxon, significance)
- [ ] J3. Can modify system live under novel constraints
- [ ] J4. Can defend every design decision
- [ ] J5. Understand trade-offs made (prompt budget vs handoff quality, cost vs coverage, etc.)

---

## GAP TRACKER

Gaps identified between plan and spec — must be addressed:

| # | Gap | Status |
|---|-----|--------|
| 1 | Rubric mutation in Phase 11 (not just meta-eval) | FIXING |
| 2 | Rewriter outputs single component — should allow multi-component mutation | FIXING |
| 3 | Score ceiling/floor detection → trigger rubric refinement | FIXING |
| 4 | System-level scoring explicitly checks handoff continuity degradation | VERIFY |
| 5 | Decision journal not in plan phases | ADDING |
| 6 | Audio recording of Agent 2 not in plan | ADDING |
| 7 | Short demo video not in plan | ADDING |
| 8 | README with detailed docs not in plan | ADDING |
| 9 | Scoring weights evolvable (not just rubric text) | FIXING |
| 10 | "Live and runnable at all times" → implies deployment, not just docker compose | ADDING |
| 11 | Temporal stage timeouts not explicit in plan/checklist | ADDING |
| 12 | Exhausted Assessment no-response path to Resolution not explicit enough | ADDING |
| 13 | Voice transcript capture quality missing from handoff checklist | ADDING |
| 14 | Stop-contact propagation across later stages/retries missing | ADDING |
| 15 | Budget enforcement evidence artifacts not explicit | ADDING |
| 16 | Fair comparison when rubrics mutate not explicit enough | ADDING |
| 17 | Evaluator gaming / objective hacking checks missing from meta-eval checklist | ADDING |
