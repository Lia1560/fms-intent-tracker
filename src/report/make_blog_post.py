from pathlib import Path
import re

STYLE = """
<style>
.chips { margin: 8px 0 12px 0; }
.chip { display: inline-block; padding: 4px 8px; margin: 3px; border: 1px solid #ddd; border-radius: 12px; font-size: 0.9em; }
</style>
"""

def _grab_section_counts(md: str):
    sections = [
        "Product & Feature Signals","Strategic Moves",
        "Regulation & Risk","Market & Trend Signals","Influencer & Analyst Commentary"
    ]
    counts = {}
    for sec in sections:
        m = re.search(rf"## {re.escape(sec)}\n((?:- .+\n?)+)", md)
        if not m:
            counts[sec] = 0
            continue
        bullets = [b for b in m.group(1).strip().splitlines() if b.strip().startswith("- ")]
        counts[sec] = len(bullets)
    return counts

def _extract_emerging(md: str):
    new, mom = [], []
    m1 = re.search(r"\*\*New this week:\*\* (.+)", md)
    if m1: new = re.findall(r"`([^`]+)`\s*\((\d+)\)", m1.group(1))
    m2 = re.search(r"\*\*Momentum:\*\* (.+)", md)
    if m2: mom = re.findall(r"`([^`]+)`\s*\(\+?(\d+)%\)", m2.group(1))
    return new, mom

def _narrative(counts, new_terms, mom_terms):
    parts = []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [f"{k} ({v})" for k,v in ranked if v > 0][:3]
    if top: parts.append("Activity concentrated in " + ", ".join(top) + ".")
    if new_terms:
        nt = [t for t,_ in new_terms[:3]]
        parts.append("New signals included " + ", ".join(nt) + ".")
    if mom_terms:
        mt = [f"{t} (+{p}%)" for t,p in mom_terms[:3]]
        parts.append("Momentum terms were " + ", ".join(mt) + ".")
    if not parts: parts.append("A quieter week with scattered updates across the market.")
    return " ".join(parts)

def _chips_block(new_terms, mom_terms):
    chips_html = ['<div class="chips">']
    if new_terms:
        chips_html.append("<strong>New this week:</strong> ")
        chips_html += [f'<span class="chip">{t}</span>' for t,_ in new_terms[:8]]
    if mom_terms:
        if new_terms: chips_html.append("<br/>")
        chips_html.append("<strong>Momentum:</strong> ")
        chips_html += [f'<span class="chip">{t}</span>' for t,_ in mom_terms[:8]]
    chips_html.append("</div>")
    return "".join(chips_html)

def write_post(report_path: Path, blog_dir: Path):
    md = report_path.read_text(encoding="utf-8")
    date = report_path.stem

    counts = _grab_section_counts(md)
    new_terms, mom_terms = _extract_emerging(md)

    title = f'Top discoveries — {date}'
    intro = _narrative(counts, new_terms, mom_terms)
    chips = _chips_block(new_terms, mom_terms) if (new_terms or mom_terms) else ""

    lines = [
        "---",
        "layout: post",
        f'title: "{title}"',
        f"date: {date}",
        "tags: [" + ",".join([t.lower().replace(' ','-') for t,c in counts.items() if c>0]) + "]",
        "---",
        STYLE,
        "",
        intro,
        chips,
        "### Highlights",
    ]

    # Include the first 2 bullets from each section
    for section in [
        "Product & Feature Signals","Strategic Moves","Regulation & Risk",
        "Market & Trend Signals","Influencer & Analyst Commentary"
    ]:
        sec_match = re.search(rf"## {re.escape(section)}\n((?:- .+\n?)+)", md)
        if not sec_match: continue
        bullets = [b.strip("- ").strip() for b in sec_match.group(1).strip().splitlines()][:2]
        if bullets:
            lines.append(f"#### {section}")
            lines += [f"- {b}" for b in bullets]
            lines.append("")

    outdir = blog_dir / "_posts"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{date}-top-discoveries.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    reports = sorted((ROOT / "reports").glob("*.md"))
    if not reports: raise SystemExit("No report found")
    latest = reports[-1]
    p = write_post(latest, ROOT / "docs")
    print(f"Wrote blog post: {p}")
