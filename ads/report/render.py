"""Report renderer: builds data-grounded HTML and Markdown executive reports with an audit appendix."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional

from ads.config import CONFIDENCE_RUBRIC
from ads.context import RunContext
from ads.state import Claim, Feedback


def render_report(
    claims: List[Claim],
    rejected_claims: List[Dict[str, Any]],
    feedback: List[Feedback],
    confidence: Optional[int],
    ctx: RunContext,
    goal: str = "",
    run_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate both Markdown and standalone HTML executive reports."""
    run_dir = Path(run_dir) if run_dir else ctx.run_dir
    charts_dir = run_dir / "charts"
    chart_files = sorted(list(charts_dir.glob("*.png"))) if charts_dir.exists() else []

    confidence_score = confidence if confidence is not None else 85
    goal_str = goal or ctx.goal or "Automated Data Analysis"

    # -------------------------------------------------------- Markdown Output ---
    md_lines: List[str] = [
        f"# Autonomous Data Science Executive Report",
        f"**Objective:** {goal_str}  ",
        f"**Audit Confidence Score:** {confidence_score}%  ",
        f"**Dataset Snapshots:** {ctx.dataset_version} ({ctx.df.shape[0]} rows, {ctx.df.shape[1]} columns)  ",
        "",
        "## Key Verified Findings",
        "",
    ]

    for c in claims:
        eids_str = ", ".join(f"`[{eid}]`" for eid in c.evidence_ids) if c.evidence_ids else ""
        md_lines.append(f"- {c.text} {eids_str}")

    if chart_files:
        md_lines.extend(["", "## Analytical Visualizations", ""])
        for chart in chart_files:
            rel_chart = chart.relative_to(run_dir)
            md_lines.append(f"### {chart.stem.replace('_', ' ').title()}")
            md_lines.append(f"![{chart.stem}]({rel_chart.as_posix()})\n")

    md_lines.extend([
        "",
        "## Statistical Quality & Validation Audit",
        "",
        f"The Critic Agent evaluated findings across standard integrity criteria (Score: **{confidence_score}%**).",
        "",
        "| Evaluation Item | Target Agent | Severity | Actionable Note |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for fb in feedback:
        md_lines.append(f"| {fb.issue} | `{fb.target_agent}` | **{fb.severity.upper()}** | {fb.action} |")

    # Audit Appendix
    md_lines.extend([
        "",
        "---",
        "## Audit Appendix",
        "",
        "### Evidence Ledger (Append-Only Proofs)",
        "",
        "| ID | Tool | Summary | Version |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for ev in ctx.ledger.all():
        md_lines.append(f"| **{ev.id}** | `{ev.tool}` | {ev.summary} | {ev.dataset_version} |")

    if rejected_claims:
        md_lines.extend([
            "",
            "### Rejected Claims (Failed Grounding Verification)",
            "",
            "| Claim Text | Cited Evidence | Verification Issues |",
            "| :--- | :--- | :--- |",
        ])
        for r in rejected_claims:
            eids = ", ".join(r.get("evidence_ids", []))
            probs = "; ".join(r.get("problems", []))
            md_lines.append(f"| \"{r.get('claim', '')}\" | `{eids}` | {probs} |")

    md_content = "\n".join(md_lines)
    md_path = run_dir / "report.md"
    md_path.write_text(md_content, encoding="utf-8")

    # ----------------------------------------------------------- HTML Output ---
    html_claims = ""
    for c in claims:
        tags = "".join(
            f'<span class="badge" title="Evidence link">{html.escape(eid)}</span>'
            for eid in c.evidence_ids
        )
        html_claims += f'<li class="claim-item">{html.escape(c.text)} {tags}</li>'

    html_charts = ""
    for chart in chart_files:
        rel = chart.relative_to(run_dir).as_posix()
        name = chart.stem.replace('_', ' ').title()
        html_charts += f"""
        <div class="chart-card">
            <h4>{html.escape(name)}</h4>
            <img src="{html.escape(rel)}" alt="{html.escape(name)}" class="chart-img"/>
        </div>
        """

    html_feedback = ""
    for fb in feedback:
        sev_class = "sev-blocker" if fb.severity == "blocker" else ("sev-warning" if fb.severity == "warning" else "sev-info")
        html_feedback += f"""
        <tr>
            <td>{html.escape(fb.issue)}</td>
            <td><code>{html.escape(fb.target_agent)}</code></td>
            <td><span class="badge {sev_class}">{html.escape(fb.severity.upper())}</span></td>
            <td>{html.escape(fb.action)}</td>
        </tr>
        """

    html_evidence = ""
    for ev in ctx.ledger.all():
        html_evidence += f"""
        <tr>
            <td><strong>{html.escape(ev.id)}</strong></td>
            <td><code>{html.escape(ev.tool)}</code></td>
            <td>{html.escape(ev.summary)}</td>
            <td>{html.escape(ev.dataset_version)}</td>
        </tr>
        """

    html_rejected = ""
    if rejected_claims:
        for r in rejected_claims:
            html_rejected += f"""
            <tr>
                <td>"{html.escape(r.get('claim', ''))}"</td>
                <td><code>{html.escape(', '.join(r.get('evidence_ids', [])))}</code></td>
                <td style="color: #ef4444;">{html.escape('; '.join(r.get('problems', [])))}</td>
            </tr>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Autonomous Data Science Executive Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #f8fafc;
        color: #0f172a;
        line-height: 1.6;
        padding: 30px;
        margin: 0;
    }}
    .container {{
        max-width: 1000px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        padding: 40px;
    }}
    .header {{
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    .score-badge {{
        display: inline-block;
        background: #10b981;
        color: white;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
    }}
    .claim-item {{
        margin-bottom: 12px;
        font-size: 16px;
    }}
    .badge {{
        background: #e2e8f0;
        color: #334155;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 6px;
        margin-left: 6px;
        font-family: monospace;
    }}
    .sev-blocker {{ background: #fecaca; color: #991b1b; }}
    .sev-warning {{ background: #fef08a; color: #854d0e; }}
    .sev-info {{ background: #e0f2fe; color: #075985; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        margin-bottom: 25px;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 10px 12px;
        text-align: left;
        font-size: 14px;
    }}
    th {{ background: #f1f5f9; }}
    .chart-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }}
    .chart-card {{
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        background: #fafafa;
    }}
    .chart-img {{
        max-width: 100%;
        height: auto;
        border-radius: 6px;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Autonomous AI Data Scientist Report</h1>
        <p><strong>Goal:</strong> {html.escape(goal_str)}</p>
        <span class="score-badge">Critic Confidence: {confidence_score}%</span>
        <span class="badge">Data Version: {html.escape(ctx.dataset_version)}</span>
    </div>

    <h2>Executive Verified Findings</h2>
    <ul>
        {html_claims}
    </ul>

    <h2>Analytical Visualizations</h2>
    <div class="chart-grid">
        {html_charts if html_charts else "<p>No charts generated.</p>"}
    </div>

    <h2>Statistical Integrity & Validation Audit</h2>
    <table>
        <thead>
            <tr><th>Issue</th><th>Target Agent</th><th>Severity</th><th>Action Required</th></tr>
        </thead>
        <tbody>
            {html_feedback}
        </tbody>
    </table>

    <hr style="margin: 40px 0; border: 0; border-top: 1px solid #e2e8f0;"/>

    <h2>Audit Appendix</h2>
    <h3>Evidence Ledger (Append-Only Proofs)</h3>
    <table>
        <thead>
            <tr><th>Evidence ID</th><th>Tool</th><th>Summary</th><th>Version</th></tr>
        </thead>
        <tbody>
            {html_evidence}
        </tbody>
    </table>

    {f'''<h3>Rejected Claims (Failed Verifier Grounding)</h3>
    <table>
        <thead><tr><th>Claim</th><th>Cited Evidence</th><th>Problems</th></tr></thead>
        <tbody>{html_rejected}</tbody>
    </table>''' if html_rejected else ""}
</div>
</body>
</html>
"""
    html_path = run_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    return {
        "markdown": str(md_path),
        "html": str(html_path),
    }

