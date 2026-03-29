#!/usr/bin/env python3
"""
Render 570-case Qwen 32B benchmark results into a human-readable HTML report.
Designed for medical student reviewers to assess diagnostic reasoning quality.

Usage:
    python3 render_benchmark_report.py
    # Outputs: benchmark_report_qwen32b.html
"""

import json
import glob
import os
import html
from datetime import datetime

RESULTS_DIR = "/tmp/qwen32b_extract/benchmark/results_qwen32b"
OUTPUT_FILE = "/Users/delcoburn/Local-DDx/benchmark_report_qwen32b.html"

ROUND_LABELS = {
    "specialized_ranking": "Round 1 — Specialized Ranking",
    "symptom_management": "Round 2 — Symptom Management",
    "team_differentials": "Round 3 — Team Differentials",
    "master_list": "Round 4 — Master List",
    "refinement": "Round 5 — Refinement",
    "voting": "Round 6 — Voting",
    "cant_miss": "Round 7 — Can't-Miss Diagnoses",
}

ROUND_DESCRIPTIONS = {
    "specialized_ranking": "Each specialist ranks the relevance of their expertise and proposes initial differential diagnoses.",
    "symptom_management": "Specialists analyze symptom patterns, onset timing, and management considerations.",
    "team_differentials": "Team collaborates to build a unified differential diagnosis list.",
    "master_list": "Consolidation of all proposed diagnoses into a ranked master list.",
    "refinement": "Critical review — specialists challenge, refine, and re-rank diagnoses.",
    "voting": "Formal Borda-weighted voting on final differential diagnoses.",
    "cant_miss": "Safety net — flag any dangerous diagnoses that must not be missed.",
}


def load_cases():
    files = sorted(glob.glob(f"{RESULTS_DIR}/case_*.json"))
    cases = []
    for f in files:
        with open(f) as fh:
            cases.append(json.load(fh))
    return cases


def esc(text):
    return html.escape(str(text))


def truncate(text, max_len=0):
    """Pass through full text (truncation disabled)."""
    return text


def render_case_card(case, idx):
    ci = case.get("case_index", idx + 1)
    case_info = case.get("case", {})
    case_name = case_info.get("name", f"case_{ci:03d}")
    description = case_info.get("description", "No description available.")
    specialty = case_info.get("specialty", "General")
    ground_truth = case.get("ground_truth", {})
    specialists = case.get("specialists", [])
    results = case.get("results", {})
    rounds = results.get("rounds", {})
    final_dx = results.get("final_diagnoses", [])
    voting = results.get("voting_result", {})
    credibility = results.get("credibility_scores", {})
    duration = case.get("duration_seconds", 0)
    config = case.get("config", {})

    gt_diagnoses = list(ground_truth.keys()) if isinstance(ground_truth, dict) else []

    # Check if any ground truth diagnosis appears in final DDx (case-insensitive)
    final_dx_lower = [d.lower() for d in final_dx]
    gt_lower = [g.lower() for g in gt_diagnoses]
    hit = any(
        any(gt_word in fdx for fdx in final_dx_lower)
        for gt_word in gt_lower
    )

    ranked = voting.get("ranked", [])

    lines = []
    lines.append(f'<div class="case-card" id="case-{ci}">')

    # Header
    status_class = "hit" if hit else "miss"
    status_label = "MATCH" if hit else "MISS"
    lines.append(
        f'<div class="case-header {status_class}" onclick="toggleCase(\'body-{ci}\')">'
    )
    lines.append(
        f'<span class="case-num">Case {ci:03d}</span>'
        f'<span class="case-specialty">{esc(specialty)}</span>'
        f'<span class="status-badge {status_class}">{status_label}</span>'
        f'<span class="duration">{duration:.0f}s</span>'
        f'<span class="toggle-icon">▶</span>'
    )
    lines.append("</div>")

    # Collapsible body
    lines.append(f'<div class="case-body" id="body-{ci}" style="display:none;">')

    # Patient presentation
    lines.append('<div class="section">')
    lines.append('<h3>Patient Presentation</h3>')
    lines.append(f'<div class="presentation">{esc(description)}</div>')
    lines.append("</div>")

    # Ground truth
    lines.append('<div class="section">')
    lines.append('<h3>Ground Truth Diagnoses</h3>')
    lines.append("<ul>")
    for dx in gt_diagnoses:
        symptoms = ground_truth[dx]
        symptom_str = ", ".join(symptoms[:5]) if isinstance(symptoms, list) and symptoms else ""
        lines.append(f"<li><strong>{esc(dx)}</strong>")
        if symptom_str:
            lines.append(f' <span class="gt-symptoms">({esc(symptom_str)})</span>')
        lines.append("</li>")
    lines.append("</ul>")
    lines.append("</div>")

    # Specialist team
    lines.append('<div class="section">')
    lines.append('<h3>Specialist Team</h3>')
    lines.append('<div class="specialists">')
    for s in specialists:
        model_tag = "Conservative" if "conservative" in s.get("model", "") else "Innovative"
        cred = credibility.get(s["name"], {})
        cred_score = cred.get("final_score", 0) if isinstance(cred, dict) else 0
        lines.append(
            f'<div class="specialist-chip">'
            f'<strong>{esc(s["name"])}</strong> — {esc(s["specialty"])} '
            f'<span class="model-tag {model_tag.lower()}">{model_tag}</span> '
            f'<span class="cred-score">Cred: {cred_score:.0f}</span>'
            f"</div>"
        )
    lines.append("</div>")
    lines.append("</div>")

    # Rounds
    lines.append('<div class="section">')
    lines.append('<h3>Diagnostic Rounds</h3>')

    for round_key, round_label in ROUND_LABELS.items():
        rd = rounds.get(round_key)
        if not rd:
            continue
        responses = rd.get("responses", [])
        rd_duration = rd.get("duration", 0)
        lines.append(f'<div class="round">')
        lines.append(
            f'<div class="round-header" onclick="toggleRound(this)">'
            f'<strong>{round_label}</strong> '
            f'<span class="round-meta">{len(responses)} responses, {rd_duration:.0f}s</span>'
            f'<span class="round-desc">{ROUND_DESCRIPTIONS.get(round_key, "")}</span>'
            f'<span class="toggle-icon">▶</span>'
            f"</div>"
        )
        lines.append(f'<div class="round-body" style="display:none;">')
        for resp in responses:
            agent = resp.get("agent_name", "Unknown")
            spec = resp.get("specialty", "")
            content = resp.get("content", "")
            conf = resp.get("confidence_score", None)
            rq = resp.get("reasoning_quality", None)

            lines.append(f'<div class="response">')
            lines.append(
                f'<div class="resp-header">'
                f'<strong>{esc(agent)}</strong> ({esc(spec)})'
            )
            if conf is not None:
                try:
                    lines.append(f' <span class="conf">Conf: {float(conf):.2f}</span>')
                except (ValueError, TypeError):
                    lines.append(f' <span class="conf">Conf: {esc(str(conf))}</span>')
            if rq is not None:
                try:
                    lines.append(f' <span class="rq">RQ: {float(rq):.2f}</span>')
                except (ValueError, TypeError):
                    lines.append(f' <span class="rq">RQ: {esc(str(rq))}</span>')
            lines.append("</div>")
            lines.append(
                f'<div class="resp-content">{esc(truncate(content))}</div>'
            )
            lines.append("</div>")
        lines.append("</div>")  # round-body
        lines.append("</div>")  # round
    lines.append("</div>")

    # Final differential
    lines.append('<div class="section">')
    lines.append('<h3>Final Differential Diagnosis</h3>')
    if ranked:
        lines.append('<table class="ddx-table"><tr><th>Rank</th><th>Diagnosis</th><th>Borda Score</th></tr>')
        for i, item in enumerate(ranked):
            dx_name = item[0] if isinstance(item, list) else item
            score = item[1] if isinstance(item, list) and len(item) > 1 else ""
            # Highlight if matches ground truth
            is_gt = any(g.lower() in dx_name.lower() for g in gt_diagnoses)
            row_class = ' class="gt-match"' if is_gt else ""
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            lines.append(
                f"<tr{row_class}><td>{i+1}</td><td>{esc(dx_name)}</td><td>{score_str}</td></tr>"
            )
        lines.append("</table>")
    else:
        lines.append("<p>No ranked diagnoses available.</p>")
    lines.append("</div>")

    lines.append("</div>")  # case-body
    lines.append("</div>")  # case-card
    return "\n".join(lines)


def compute_summary(cases):
    total = len(cases)
    durations = [c.get("duration_seconds", 0) for c in cases]
    avg_dur = sum(durations) / total if total else 0

    hits = 0
    for c in cases:
        gt = c.get("ground_truth", {})
        gt_names = [g.lower() for g in gt.keys()] if isinstance(gt, dict) else []
        final_dx = c.get("results", {}).get("final_diagnoses", [])
        final_lower = [d.lower() for d in final_dx]
        if any(any(g in f for f in final_lower) for g in gt_names):
            hits += 1

    # Specialty distribution
    spec_counts = {}
    for c in cases:
        spec = c.get("case", {}).get("specialty", "Unknown")
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    return {
        "total": total,
        "hits": hits,
        "miss": total - hits,
        "hit_rate": hits / total * 100 if total else 0,
        "avg_duration": avg_dur,
        "total_duration_hours": sum(durations) / 3600,
        "specialties": sorted(spec_counts.items(), key=lambda x: -x[1]),
    }


def render_html(cases):
    summary = compute_summary(cases)

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local-DDx Benchmark Report — Qwen 32B (570 cases)</title>
<style>
:root {{
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --muted: #6c757d;
    --hit: #28a745;
    --miss: #dc3545;
    --accent: #0d6efd;
    --round-bg: #f1f3f5;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 20px;
    max-width: 1100px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
h2 {{ font-size: 1.3rem; margin: 20px 0 10px; border-bottom: 2px solid var(--accent); padding-bottom: 4px; }}
h3 {{ font-size: 1.05rem; margin: 12px 0 6px; color: var(--accent); }}

.report-header {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}}
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 12px 0;
}}
.stat-box {{
    background: var(--round-bg);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}}
.stat-box .value {{ font-size: 1.6rem; font-weight: 700; }}
.stat-box .label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; }}
.stat-box.hit .value {{ color: var(--hit); }}
.stat-box.miss .value {{ color: var(--miss); }}

.controls {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}}
.controls input, .controls select {{
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.9rem;
}}
.controls input {{ flex: 1; min-width: 200px; }}
.controls button {{
    padding: 6px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--accent);
    color: white;
    cursor: pointer;
    font-size: 0.9rem;
}}

.case-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}}
.case-header {{
    padding: 10px 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    user-select: none;
    transition: background 0.15s;
}}
.case-header:hover {{ background: var(--round-bg); }}
.case-header.hit {{ border-left: 4px solid var(--hit); }}
.case-header.miss {{ border-left: 4px solid var(--miss); }}
.case-num {{ font-weight: 700; min-width: 75px; }}
.case-specialty {{ color: var(--muted); font-size: 0.85rem; min-width: 120px; }}
.status-badge {{
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
}}
.status-badge.hit {{ background: #d4edda; color: var(--hit); }}
.status-badge.miss {{ background: #f8d7da; color: var(--miss); }}
.duration {{ color: var(--muted); font-size: 0.8rem; margin-left: auto; }}
.toggle-icon {{ color: var(--muted); font-size: 0.7rem; }}

.case-body {{ padding: 0 20px 20px; }}
.section {{ margin-bottom: 16px; }}
.presentation {{
    background: var(--round-bg);
    padding: 14px;
    border-radius: 6px;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
}}

.specialists {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.specialist-chip {{
    background: var(--round-bg);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
}}
.model-tag {{
    font-size: 0.7rem;
    padding: 1px 6px;
    border-radius: 8px;
    font-weight: 600;
}}
.model-tag.conservative {{ background: #cce5ff; color: #004085; }}
.model-tag.innovative {{ background: #fff3cd; color: #856404; }}
.cred-score {{ font-size: 0.75rem; color: var(--muted); }}

.round {{
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 6px;
}}
.round-header {{
    padding: 8px 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    background: var(--round-bg);
    flex-wrap: wrap;
}}
.round-header:hover {{ background: #e9ecef; }}
.round-meta {{ font-size: 0.75rem; color: var(--muted); }}
.round-desc {{ font-size: 0.78rem; color: var(--muted); font-style: italic; margin-left: auto; }}
.round-body {{ padding: 10px 14px; }}

.response {{
    border-bottom: 1px solid var(--border);
    padding: 8px 0;
}}
.response:last-child {{ border-bottom: none; }}
.resp-header {{ font-size: 0.88rem; margin-bottom: 4px; }}
.conf, .rq {{
    font-size: 0.75rem;
    background: var(--round-bg);
    padding: 1px 6px;
    border-radius: 8px;
    margin-left: 4px;
}}
.resp-content {{
    font-size: 0.85rem;
    color: #495057;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
    line-height: 1.5;
}}

.ddx-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
.ddx-table th {{
    background: var(--round-bg);
    text-align: left;
    padding: 6px 10px;
    border-bottom: 2px solid var(--border);
}}
.ddx-table td {{
    padding: 5px 10px;
    border-bottom: 1px solid var(--border);
}}
.ddx-table tr.gt-match {{
    background: #d4edda;
    font-weight: 600;
}}

.gt-symptoms {{ font-size: 0.8rem; color: var(--muted); }}

.specialty-table {{
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 8px 0;
}}
.specialty-table td, .specialty-table th {{
    padding: 4px 12px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}}

@media print {{
    .controls {{ display: none; }}
    .case-body {{ display: block !important; }}
    .round-body {{ display: block !important; }}
    body {{ font-size: 10pt; }}
}}
</style>
</head>
<body>

<div class="report-header">
<h1>Local-DDx Benchmark Report</h1>
<p>Qwen 2.5 32B Instruct (Q8) &mdash; {summary['total']} cases from Open-XDDx</p>
<p style="color:var(--muted); font-size:0.85rem;">Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} &bull; 7-round pipeline with Borda voting &amp; credibility scoring</p>

<div class="summary-grid">
    <div class="stat-box"><div class="value">{summary['total']}</div><div class="label">Total Cases</div></div>
    <div class="stat-box hit"><div class="value">{summary['hits']}</div><div class="label">Matches</div></div>
    <div class="stat-box miss"><div class="value">{summary['miss']}</div><div class="label">Misses</div></div>
    <div class="stat-box hit"><div class="value">{summary['hit_rate']:.1f}%</div><div class="label">Rough Match Rate</div></div>
    <div class="stat-box"><div class="value">{summary['avg_duration']:.0f}s</div><div class="label">Avg Duration</div></div>
    <div class="stat-box"><div class="value">{summary['total_duration_hours']:.1f}h</div><div class="label">Total Compute</div></div>
</div>

<p style="font-size:0.8rem; color:var(--muted); margin-top:8px;">
<strong>Note:</strong> "Match" = substring match between ground truth and final DDx (rough heuristic).
For precise recall/precision, use the benchmark evaluator with synonym mapping. Green-highlighted rows in final DDx tables indicate ground-truth matches.
</p>

<h2>Cases by Specialty</h2>
<table class="specialty-table">
<tr><th>Specialty</th><th>Count</th></tr>
""")
    for spec, count in summary["specialties"]:
        html_parts.append(f"<tr><td>{esc(spec)}</td><td>{count}</td></tr>")
    html_parts.append("</table></div>")

    # Controls
    html_parts.append("""
<div class="controls">
    <input type="text" id="searchBox" placeholder="Search cases (diagnosis, specialty, case number)..." oninput="filterCases()">
    <select id="statusFilter" onchange="filterCases()">
        <option value="all">All</option>
        <option value="hit">Matches only</option>
        <option value="miss">Misses only</option>
    </select>
    <button onclick="expandAll()">Expand All</button>
    <button onclick="collapseAll()">Collapse All</button>
</div>
""")

    # Render each case
    html_parts.append('<div id="caseContainer">')
    for i, case in enumerate(cases):
        html_parts.append(render_case_card(case, i))
    html_parts.append("</div>")

    # JavaScript
    html_parts.append("""
<script>
function toggleCase(id) {
    const el = document.getElementById(id);
    const icon = el.previousElementSibling.querySelector('.toggle-icon');
    if (el.style.display === 'none') {
        el.style.display = 'block';
        icon.textContent = '▼';
    } else {
        el.style.display = 'none';
        icon.textContent = '▶';
    }
}
function toggleRound(header) {
    const body = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');
    if (body.style.display === 'none') {
        body.style.display = 'block';
        icon.textContent = '▼';
    } else {
        body.style.display = 'none';
        icon.textContent = '▶';
    }
}
function expandAll() {
    document.querySelectorAll('.case-body').forEach(el => { el.style.display = 'block'; });
    document.querySelectorAll('.round-body').forEach(el => { el.style.display = 'block'; });
    document.querySelectorAll('.toggle-icon').forEach(el => { el.textContent = '▼'; });
}
function collapseAll() {
    document.querySelectorAll('.case-body').forEach(el => { el.style.display = 'none'; });
    document.querySelectorAll('.round-body').forEach(el => { el.style.display = 'none'; });
    document.querySelectorAll('.toggle-icon').forEach(el => { el.textContent = '▶'; });
}
function filterCases() {
    const query = document.getElementById('searchBox').value.toLowerCase();
    const status = document.getElementById('statusFilter').value;
    document.querySelectorAll('.case-card').forEach(card => {
        const header = card.querySelector('.case-header');
        const text = card.textContent.toLowerCase();
        const isHit = header.classList.contains('hit');
        const matchesQuery = !query || text.includes(query);
        const matchesStatus = status === 'all' || (status === 'hit' && isHit) || (status === 'miss' && !isHit);
        card.style.display = (matchesQuery && matchesStatus) ? 'block' : 'none';
    });
}
</script>
</body>
</html>
""")
    return "\n".join(html_parts)


def main():
    print("Loading cases...")
    cases = load_cases()
    print(f"Loaded {len(cases)} cases.")

    print("Rendering HTML...")
    html_content = render_html(cases)

    with open(OUTPUT_FILE, "w") as f:
        f.write(html_content)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Done! Written to {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
