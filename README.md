# Darwin Voice Agent

Self-learning debt collection system with 3 AI agents, Temporal orchestration, and Darwin Godel Machine (DGM) meta-evaluation.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
# Chat UI (Agent 1 & 3)
chainlit run interfaces/chat/app.py --port 8000

# Voice server (Agent 2)
python -m interfaces.voice.call_server

# Evolution dashboard
python -m interfaces.dashboard.server

# Run evolution loop
python main.py evolve --max-gen 5

# Run 10 parallel evolution runs
python tools/run_parallel.py --n-runs 10 --max-gen 3
```

## Docker

```bash
docker compose up --build
# Temporal UI: localhost:8080 | Chat: localhost:8000 | Voice: localhost:8001 | Dashboard: localhost:8050
```

## Architecture

```
Borrower → Agent 1 (Chat) → Handoff → Agent 2 (Voice) → Handoff → Agent 3 (Chat)
```

**Single code path:** `agents/core.py::agent_respond()` — used by simulation, Temporal, Chainlit, and Pipecat.

**Token budgets:** 2000 tokens/agent, 500 tokens/handoff. Enforced in code, evidenced in logs.

**Temporal workflow:** One workflow per borrower. Assessment (retry max 3) → Resolution (deal? exit) → Final Notice (resolve or flag).

## Self-Learning Loop

1. **Mutate** — GEPA-style rewriter with full conversation evidence, TextGrad feedback, crossover, few-shot examples
2. **Evaluate** — Staged: 2 → 10 → 34 conversations across 8 personas (including adversarial)
3. **Compare** — Paired bootstrap CI (1000 resamples, 95%) + Wilcoxon. Variance-aware.
4. **Meta-eval** — Every 4 generations: rewrite miscalibrated rubrics, adjust weights. Compliance rules immutable.

## Scoring (per conversation)

| Category | Weight | Checks |
|----------|--------|--------|
| Goal completion | 27% | 7-9 per agent |
| Compliance | 23% | 8 rules (binary gate) |
| Quality | 18% | 7 per agent |
| Handoff | 13% | 8 per handoff |
| System continuity | 9% | 7 cross-agent |
| Deal quality | 10% | Best deal for company |

## Compliance Rules (Immutable)

1. AI identity disclosure  2. No false threats  3. Honor stop-contact  4. Settlement within policy  5. Hardship program offered  6. Recording disclosure  7. Professional composure  8. Data privacy

## Reproducibility

```bash
# Re-score stored transcripts
python main.py rerun-eval --config results/run2/meta.json --sample 10
```

Raw data: `results/run2/` — 705 transcripts, archive.json, costs.json, lessons.jsonl

## Project Structure

```
agents/          Core: agent_respond(), prompts, strategy
config.py        Settings singleton
data/            Mock borrower DB (7 profiles)
evaluation/      Scoring, compliance, deal quality, stats
evolution/       DGM loop, rewriter, meta-eval, archive
handoff/         Cross-modal summarization (≤500 tokens)
interfaces/      Chat (Chainlit), Voice (Pipecat), Dashboard
models/          Domain types (enums, domain, scoring, cost)
simulation/      Test harness: 8 personas, pipeline simulator
tools/           Parallel runs, report generation
workflow/        Temporal: activities, pipeline, worker
tests/           34 tests
```
