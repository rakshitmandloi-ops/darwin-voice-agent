# Evolution Report

## Run 2 — Longest Run (8 Generations, 2 Meta-Eval Cycles)

| Metric | Value |
|--------|-------|
| Seed score | 6.706 |
| Best score | 7.264 (v5_3_b99e, gen 5) |
| Improvement | +0.558 (+8.3%) |
| Variants | 29 total, 19 active, 10 discarded |
| Conversations | 705 |
| Cost | $11.33 / $20.00 |

### Cost Breakdown

| Category | Cost | Calls | % |
|----------|------|-------|---|
| Simulation | $7.81 | 29,276 | 69% |
| Evaluation | $2.50 | 31,087 | 22% |
| Summarization | $0.47 | 1,313 | 4% |
| Rewriting | $0.54 | 32 | 5% |
| Meta-eval | $0.01 | 1 | <1% |

### Per-Agent Prompt Evolution

Components modified across 29 variants:
- agent2_prompt: 24 times
- agent1_prompt: 4 times
- summarizer_prompt: 1 time
- agent3_prompt: 0 times

### Evolution Trajectory

| Gen | Variant | Score | Parent | Convos | Status | Component |
|-----|---------|-------|--------|--------|--------|-----------|
| 0 | v0 | 6.71 | - | 5 | seed | - |
| 1 | v1_1_98ab | 5.40 | v0 | 2 | DISCARDED | agent2_prompt |
| 1 | v1_3_1f4d | 6.01 | v0 | 2 | DISCARDED | agent2_prompt |
| 1 | v1_2_2c79 | 6.73 | v0 | 25 | active | agent2_prompt |
| 1 | v1_0_e718 | 7.11 | v0 | 25 | active | agent2_prompt |
| 2 | v2_2_247d | 5.37 | v1_2_2c79 | 2 | DISCARDED | agent2_prompt |
| 2 | v2_3_1b7e | 6.63 | v1_0_e718 | 25 | active | agent2_prompt |
| 2 | v2_0_da6f | 6.79 | v1_2_2c79 | 25 | active | agent2_prompt |
| 2 | v2_1_f75c | 7.01 | v1_0_e718 | 25 | active | agent2_prompt |
| 3 | v3_1_0d9f | 5.61 | v2_3_1b7e | 2 | DISCARDED | agent2_prompt |
| 3 | v3_2_43c4 | 7.02 | v2_3_1b7e | 25 | active | agent2+summarizer |
| 3 | v3_0_f0e0 | 7.04 | v2_0_da6f | 25 | active | agent2_prompt |
| 4 | v4_3_e161 | 6.03 | v3_0_f0e0 | 2 | DISCARDED | agent2_prompt |
| 4 | v4_1_8653 | 6.64 | v2_1_f75c | 25 | active | agent2_prompt |
| 4 | v4_2_e9d0 | 6.75 | v2_1_f75c | 25 | active | agent2_prompt |
| 4 | v4_0_6a2b | 6.58 | v2_1_f75c | 25 | active | agent1_prompt |
| 5 | v5_2_3a01 | 6.22 | v3_0_f0e0 | 2 | DISCARDED | agent2_prompt |
| 5 | v5_1_73f8 | 5.57 | v4_2_e9d0 | 2 | DISCARDED | agent1_prompt |
| 5 | v5_0_6e3d | 7.00 | v2_0_da6f | 25 | active | agent2_prompt |
| 5 | **v5_3_b99e** | **7.26** | v3_0_f0e0 | 25 | **BEST** | agent2_prompt |
| 6 | v6_3_06b8 | 6.76 | v5_0_6e3d | 25 | active | agent2_prompt |
| 6 | v6_1_651f | 6.47 | v2_0_da6f | 25 | active | agent2_prompt |
| 6 | v6_2_eb86 | 6.80 | v2_0_da6f | 25 | active | agent2_prompt |
| 6 | v6_0_3700 | 6.85 | v2_0_da6f | 25 | active | agent2_prompt |
| 7 | v7_3_cdb4 | 4.86 | v5_0_6e3d | 2 | DISCARDED | agent2_prompt |
| 7 | v7_2_72d0 | 5.08 | v3_2_43c4 | 2 | DISCARDED | agent1_prompt |
| 7 | v7_0_c4ae | 6.68 | v4_1_8653 | 25 | active | agent1_prompt |
| 7 | v7_1_b06b | 7.08 | v4_1_8653 | 25 | active | agent2_prompt |
| 8 | v8_3_356e | 5.58 | v4_0_6a2b | 2 | DISCARDED | agent2_prompt |

### Regressions Detected

10 variants discarded (34% discard rate):
- 7 discarded at Stage 1 (2 conversations) — early exit saved budget
- Score range of discarded: 4.86 to 6.22
- Most common reason: system continuity regression when changing agent2's handoff handling

### Meta-Evaluation (2 Cycles)

**Cycle 1 (Gen 0) — confidence: high, applied: yes**

Findings:
- `system/coherent_continuation`: 0% pass rate across ALL variants — criterion was miscalibrated, not agents
- `quality/agent1/concise`: 3% pass rate — criterion too strict

Rewrites:
- `quality/agent1/concise`: "(hardcoded default)" → "Ensure responses are succinct and avoid unnecessary repetition, while still providing all necessary information"
- `system/coherent_continuation`: "(hardcoded default)" → "Ensure conversation flows logically from one point to the next, maintaining context and relevance"

**Cycle 2 (Gen 12) — confidence: high, applied: yes**

Findings:
- `system/no_re_verification`: 0% pass rate — agents WERE re-verifying (a real problem, but also criterion was too strict)
- `quality/agent1/concise`: 4% (still near-floor after first rewrite)
- `system/no_re_introductions`: 3%

Rewrites:
- `system/no_re_verification`: "Agent 2 and 3 do not re-verify borrower identity" → "Avoid unnecessary re-verification unless explicitly requested by borrower"
- `system/no_re_introductions`: "No agent re-introduces itself after Agent 1" → "Agents do not reintroduce themselves unnecessarily after initial introduction"
- `quality/agent1/concise`: second rewrite for clarity

---

## Enhanced Run — With GEPA/TextGrad/Crossover (2 Generations)

| Metric | Value |
|--------|-------|
| Seed score | 7.51 |
| Variants | 6 |
| Conversations | 192 |
| Lessons generated | 4 |

### Seed Per-Persona Scores (8 personas)

| Persona | Score |
|---------|-------|
| cooperative | 8.76 |
| evasive | 8.53 |
| distressed | 8.01 |
| confused | 7.91 |
| combative | 7.78 |
| litigious | 7.46 |
| manipulative | 6.08 |
| prompt_injection | 5.58 |

Higher seed score (7.51 vs 6.71) due to agent-aware borrower context fix (Agent 2/3 no longer re-verify identity).

### Trajectory

| Gen | Variant | Score | Convos | Status | Component |
|-----|---------|-------|--------|--------|-----------|
| 0 | v0 | 7.51 | 8 | seed | - |
| 1 | v1_0_9af5 | 5.08 | 2 | DISCARDED | agent2_prompt |
| 1 | v1_1_24e2 | 7.03 | 34 | active | agent2_prompt |
| 1 | v1_3_dda0 | 7.20 | 34 | active | agent2_prompt |
| 1 | v1_2_5825 | 7.14 | 34 | active | agent2_prompt |
| 2 | v2_3_8561 | 6.02 | 2 | DISCARDED | agent1_prompt |

Gen 2 targeted **agent1_prompt** — escalation rule working (agent2 was changed 3+ times in Gen 1 without beating seed).

### Lessons Generated

```
Gen 1: "system continuity critically low, failure in coherent continuation" → not promoted, score diff -0.48
Gen 1: "system continuity issues with how agents use handoff context" → not promoted, score diff -0.31
Gen 1: "system continuity lowest at 3.41, significant failure" → DISCARDED (compliance fail)
Gen 1: "agents not effectively using handoff context" → not promoted, score diff -0.37
```

### Key Difference from Run 2

| Aspect | Run 2 (basic) | Enhanced run |
|--------|--------------|--------------|
| Rewriter input | 3 convos, 200 chars/msg | ALL convos, zero truncation |
| Lessons | None | GEPA-style structured insights |
| Crossover | None | 25% chance merge two parents |
| Feedback | Just failing criteria names | TextGrad: "agent said X, should said Y" |
| Few-shot | None | Best 3 conversations as examples |
| Escalation | None | Forced component rotation after 3 fails |
| Personas | 5 | 8 (+ manipulative, litigious, prompt_injection) |
| Deal quality | Not scored | 10% weight |
| Rewriter cost | $0.03/call | $0.17/call |

---

## Raw Data

All raw data in `results/run2/`:
- `archive.json` — all 29 variant configs + per-conversation scores
- `evolution_summary.json` — aggregated summary
- `costs.json` — per-call cost log (JSONL)
- `token_budgets.json` — budget enforcement evidence
- `meta_eval_log.json` — meta-eval decisions with before/after diffs
- `transcripts/` — 705 full conversation JSONs

Rerun evaluation:
```
python main.py rerun-eval --config results/run2/meta.json --sample 10
```
