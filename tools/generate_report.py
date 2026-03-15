"""
Generate EVOLUTION_REPORT.md from results/run2/ data.

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent / "results" / "run2"
    output_path = Path(__file__).resolve().parent.parent / "EVOLUTION_REPORT.md"

    # Load data
    with open(run_dir / "evolution_summary.json") as f:
        summary = json.load(f)

    with open(run_dir / "meta_eval_log.json") as f:
        meta_eval_entries = [json.loads(line) for line in f if line.strip()]

    # Build report
    lines: list[str] = []
    lines.append("# Evolution Report")
    lines.append("")
    lines.append(f"**Run ID:** {summary['run_id']}")
    lines.append(f"**Batch ID:** {summary['batch_id']}")
    lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Seed score | {summary['seed_score']:.3f} |")
    lines.append(f"| Best score | {summary['best_score']:.3f} |")
    lines.append(f"| Improvement | +{summary['improvement']:.3f} ({summary['improvement_pct']}%) |")
    lines.append(f"| Best variant | `{summary['best_id']}` |")
    lines.append(f"| Total variants | {summary['total_variants']} |")
    lines.append(f"| Active | {summary['active']} |")
    lines.append(f"| Discarded | {summary['discarded']} |")
    lines.append(f"| Generations | {summary['generations']} |")
    lines.append(f"| Conversations simulated | {summary['transcripts_count']} |")
    lines.append("")

    # Trajectory table
    lines.append("## Evolution Trajectory")
    lines.append("")
    lines.append("| Gen | Variant | Score | Parent | Convos | Status | Components | Mutation Summary |")
    lines.append("|-----|---------|-------|--------|--------|--------|------------|------------------|")

    for gen_str in sorted(summary["trajectory"].keys(), key=int):
        for v in summary["trajectory"][gen_str]:
            status = "DISCARDED" if v["discarded"] else "active"
            if v["id"] == summary["best_id"]:
                status = "**BEST**"
            components = ", ".join(v["components"]) if v["components"] else "-"
            mutation = v["mutation"][:80] + "..." if len(v["mutation"]) > 80 else v["mutation"]
            lines.append(
                f"| {gen_str} | `{v['id']}` | {v['score']:.2f} | "
                f"`{v['parent'] or '-'}` | {v['convos']} | {status} | "
                f"{components} | {mutation} |"
            )
    lines.append("")

    # Cost breakdown
    lines.append("## Cost Breakdown")
    lines.append("")
    cost = summary["cost"]
    lines.append(f"**Total cost:** ${cost['total_usd']:.3f}")
    lines.append(f"**Total API calls:** {cost['total_calls']:,}")
    lines.append("")
    lines.append("| Category | Cost | Calls | % of Total |")
    lines.append("|----------|------|-------|------------|")
    for cat, data in sorted(cost["by_category"].items()):
        pct = (data["cost"] / cost["total_usd"] * 100) if cost["total_usd"] > 0 else 0
        lines.append(f"| {cat} | ${data['cost']:.4f} | {data['calls']:,} | {pct:.1f}% |")
    lines.append("")

    # Meta-eval
    lines.append("## Meta-Evaluation")
    lines.append("")
    meta = summary["meta_eval"]
    lines.append(f"**Total runs:** {meta['total_runs']}")
    lines.append("")

    for i, run in enumerate(meta["runs"]):
        lines.append(f"### Meta-Eval Run {i + 1} (Generation {run['generation']})")
        lines.append("")
        lines.append(f"**Confidence:** {run['confidence']}")
        lines.append(f"**Applied:** {run['applied']}")
        lines.append("")
        lines.append("**Findings:**")
        for finding in run["findings"]:
            lines.append(f"- {finding}")
        lines.append("")

        if run.get("changes_applied", {}).get("rubric_overrides"):
            lines.append("**Rubric Rewrites:**")
            lines.append("")
            for check, change in run["changes_applied"]["rubric_overrides"].items():
                lines.append(f"- `{check}`:")
                lines.append(f"  - Before: {change['before']}")
                lines.append(f"  - After: {change['after']}")
            lines.append("")

    # Second meta-eval from the JSONL log (if more than in summary)
    if len(meta_eval_entries) > len(meta["runs"]):
        for entry in meta_eval_entries[len(meta["runs"]):]:
            lines.append(f"### Meta-Eval Run (Generation {entry['generation']})")
            lines.append("")
            lines.append(f"**Confidence:** {entry['confidence']}")
            lines.append(f"**Applied:** {entry['applied']}")
            lines.append("")
            lines.append("**Findings:**")
            for finding in entry["findings"]:
                lines.append(f"- {finding}")
            lines.append("")
            if entry.get("changes_applied", {}).get("rubric_overrides"):
                lines.append("**Rubric Rewrites:**")
                for check, change in entry["changes_applied"]["rubric_overrides"].items():
                    lines.append(f"- `{check}`:")
                    lines.append(f"  - Before: {change['before']}")
                    lines.append(f"  - After: {change['after']}")
                lines.append("")

    # Features
    lines.append("## Features Demonstrated")
    lines.append("")
    for feature in summary.get("features", []):
        lines.append(f"- {feature}")
    lines.append("")

    # Statistical rigor
    lines.append("## Statistical Rigor")
    lines.append("")
    lines.append("- Paired bootstrap CI with 1000 resamples (95% confidence)")
    lines.append("- Wilcoxon signed-rank test as secondary check")
    lines.append("- Persona-stratified comparison controls for borrower behavior variance")
    lines.append("- High-variance flag: std dev > 40% of mean diff = unreliable")
    lines.append("- Staged evaluation: 2 -> 10 -> 25 conversations (early exit for bad variants)")
    lines.append("- Strict grader (gpt-4o) validates top candidates independently")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("All data artifacts for this run are in `results/run2/`:")
    lines.append("")
    lines.append("- `archive.json` — Full archive with all variant configs and scores")
    lines.append("- `evolution_summary.json` — Aggregated summary (source for this report)")
    lines.append("- `costs.json` — Per-call cost log (JSONL)")
    lines.append("- `token_budgets.json` — Token budget evidence (JSONL)")
    lines.append("- `meta_eval_log.json` — Meta-evaluation decisions (JSONL)")
    lines.append("- `transcripts/` — All 705 conversation transcripts")
    lines.append("")
    lines.append("To re-score conversations:")
    lines.append("```bash")
    lines.append("python main.py rerun-eval --config results/run2/meta.json --sample 10")
    lines.append("```")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    print(f"Seed score: {summary['seed_score']}, Best: {summary['best_score']}, Improvement: {summary['improvement_pct']}%")


if __name__ == "__main__":
    main()
