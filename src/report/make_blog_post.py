from pathlib import Path
import re

def _load_latest_report(root: Path):
    print(">>> [make_blog_post] Loading latest report")
    reports = sorted((root / "reports").glob("*.md"))
    if not reports:
        raise SystemExit("No report found")
    md = reports[-1].read_text(encoding="utf-8")
    date = reports[-1].stem
    print(f">>> [make_blog_post] Found report: {reports[-1]}")
    return date, md

def _parse_tldr(md):
    print(">>> [make_blog_post] Parsing TL;DR")
    m = re.search(r"## TL;DR\n(.+?)(?:\n## |\Z)", md, flags=re.S)
    if not m:
        return []
    lines = [l.strip("- ").strip() for l in m.group(1).strip().splitlines() if l.strip()]
    return lines

def write_post(root: Path):
    date, md = _load_latest_report(root)
    tldr = _parse_tldr(md)

    outdir = root / "docs" / "_posts"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{date}-top-discoveries.md"

    print(f">>> [make_blog_post] Writing blog post to {out}")
    out.write_text("\n".join([
        "---",
        "layout: post",
        f'title: "Top discoveries — {date}"',
        f"date: {date}",
        "tags: [highlights,fms]",
        "---",
        "\n".join([f"- {line}" for line in tldr]) if tldr else "_No highlights available._"
    ]), encoding="utf-8")
    return out

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    print(f">>> [make_blog_post] Resolved ROOT = {ROOT}")
    p = write_post(ROOT)
    print(f">>> [make_blog_post] Wrote blog post: {p}")
