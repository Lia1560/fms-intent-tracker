from pathlib import Path
import re
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer   # ✅ correct tokenizer import
from sumy.utils import get_stop_words

def _summ(text, sents=4, lang="english"):
    # ✅ use Sumy's Tokenizer instead of the missing sumy.nltk_tokenize
    parser = PlaintextParser.from_string(text, Tokenizer(lang))
    summ = TextRankSummarizer()
    summ.stop_words = get_stop_words(lang)
    return " ".join(str(s) for s in summ(parser.document, sents))

def _load_latest_report(root: Path):
    reports = sorted((root / "reports").glob("*.md"))
    if not reports: raise SystemExit("No report found")
    md = reports[-1].read_text(encoding="utf-8")
    date = reports[-1].stem
    return date, md

def _parse_top10(md):
    m = re.search(r"## Top 10 News Items\n(.+?)(?:\n## |\Z)", md, flags=re.S)
    items = []
    if not m: return items
    chunk = m.group(1).strip()
    blocks = [b.strip() for b in chunk.split("\n\n") if b.strip()]
    for k in range(0, len(blocks), 3):
        try:
            title = re.sub(r"^\*\*\d+\.\s*","", blocks[k]).strip("* ").strip()
            desc  = blocks[k+1].strip()
            linkm = re.search(r"\[Source\]\(([^)]+)\)", blocks[k+2])
            link = linkm.group(1) if linkm else ""
            impactm = re.search(r"Impact:\s*(\w+)", blocks[k+2])
            impact = impactm.group(1) if impactm else "Medium"
            items.append({"title": title, "desc": desc, "url": link, "impact": impact})
        except Exception:
            pass
    return items[:5]

def write_post(root: Path):
    date, md = _load_latest_report(root)
    top = _parse_top10(md)

    # Section counts for intro colour
    sec_counts = {}
    for sec in ["Product & Feature Signals","Strategic Moves","Regulation & Risk","Market & Trend Signals","Influencer & Analyst Commentary"]:
        mm = re.search(rf"## {re.escape(sec)}\n((?:- .+\n?)+)", md)
        sec_counts[sec] = len(mm.group(1).strip().splitlines()) if mm else 0
    hottest = ", ".join([f"{k} ({v})" for k,v in sorted(sec_counts.items(), key=lambda kv:(-kv[1],kv[0])) if v>0][:3]) or "a quieter mix"

    # Emerging chips (optional)
    chips = ""
    m1 = re.search(r"\*\*New this week:\*\* (.+)", md); m2 = re.search(r"\*\*Momentum:\*\* (.+)", md)
    if m1 or m2:
        chips = "<p><em>New:</em> " + (m1.group(1) if m1 else "") + " "
        chips += " <em>Momentum:</em> " + (m2.group(1) if m2 else "") + "</p>"

    paras = []
    intro = (f"This week’s FMS and freelance-economy signals clustered around {hottest}. "
             f"We saw a mix of feature rollouts, strategic moves, and policy updates shaping how enterprises source, "
             f"onboard, and manage freelancers at scale.")
    paras.append(intro)

    for item in top:
        why = "This matters because it touches one of the core FMS capabilities and shifts risk, cost, or speed for enterprise programs."
        body = _summ(item["desc"], sents=3)
        paras.append(f"**{item['title']}** — {item['impact']} impact. {body} {why} [Source]({item['url']}).")

    outro = ("Stepping back, these updates reinforce 2025 themes we’re tracking: "
             "AI-assisted workflows, tighter compliance (IR35/IRS/EU), deeper integrations, "
             "and the rise of niche platforms serving specific talent communities.")
    paras.append(outro)

    text = "\n\n".join(paras)
    import re as _re
    words = len(_re.findall(r"\w+", text))
    if words < 600:
        text += "\n\n" + "In short, momentum continues to build across the FMS stack as vendors and buyers align on automation, compliance, and integrated workflows."
    elif words > 820:
        text = " ".join(text.split()[:820])

    outdir = root / "docs" / "_posts"; outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{date}-top-discoveries-longform.md"
    out.write_text("\n".join([
        "---","layout: post",f'title: "Top 5 — narrative brief — {date}"',f"date: {date}","tags: [longform,fms]","---",text,chips
    ]), encoding="utf-8")
    return out

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    p = write_post(ROOT)
    print(f"Wrote longform blog post: {p}")
