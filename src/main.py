import os, json, yaml, csv, re, time
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import defaultdict

import feedparser, requests, tldextract
from bs4 import BeautifulSoup
import trafilatura

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser

from sumy.utils import get_stop_words

from dateutil import parser as dtparser, tz
import re

def sumy_tokenize(text: str):
    """Lightweight sentence splitter to replace the missing sumy import."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[\{])', text)
    return [p.strip() for p in parts if p.strip()]

import os, socket
socket.setdefaulttimeout(int(os.getenv("HTTP_TIMEOUT", "20")))  


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SRC = yaml.safe_load((ROOT / "sources.yaml").read_text())
HIST_PATH = ROOT / "data" / "history.json"
DISC_PATH = ROOT / "data" / "discovered_sources.yaml"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(parents=True, exist_ok=True)

HIST = json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else {"terms":{}, "sources":{}}
DISC = yaml.safe_load(DISC_PATH.read_text()) if DISC_PATH.exists() else {"feeds":{}, "pending":{}}

# ---- Embeddings model (small) ----
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Utilities ----
def normalize_text(s): return re.sub(r"\s+", " ", (s or "")).strip()

def domain(u):
    try:
        ext = tldextract.extract(u)
        return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except: return urlparse(u).netloc

def is_recent(published_str, days=7, tzname="Europe/London"):
    try:
        dt = dtparser.parse(published_str)
    except Exception:
        return True
    tzinfo = tz.gettz(tzname)
    now = datetime.now(tzinfo)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=tzinfo)
    return (now - dt) <= timedelta(days=days)

def requests_get(url, timeout=12):
    headers = {"User-Agent":"Mozilla/5.0 (FMS-Intent-Tracker)"}
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

# ---- Finder: domain → feed resolution (lightweight) ----
COMMON_FEED_PATHS = ["/feed", "/rss", "/rss.xml", "/atom.xml", "/news/rss", "/blog/rss", "/press/rss", "/changelog.xml"]

def discover_feeds_for_domain(dom):
    found = set()
    try:
        # 1) Try common paths
        for p in COMMON_FEED_PATHS:
            try:
                url = f"https://{dom}{p}"
                fp = feedparser.parse(url)
                if fp.bozo == 0 and fp.entries:
                    found.add(url)
            except Exception:
                pass
        # 2) Scan homepage for <link rel="alternate" type="application/rss+xml">
        try:
            res = requests_get(f"https://{dom}")
            if res.ok:
                soup = BeautifulSoup(res.text, "html.parser")
                for link in soup.find_all("link", {"rel":"alternate"}):
                    if "rss" in (link.get("type") or "") or "atom" in (link.get("type") or ""):
                        href = link.get("href")
                        if href and href.startswith("http"):
                            fp = feedparser.parse(href)
                            if fp.bozo == 0 and fp.entries:
                                found.add(href)
        except Exception:
            pass
    except Exception:
        pass
    return list(found)[:3]  # cap

# ---- Fetch feeds ----
def fetch_feed(url):
    feed = feedparser.parse(url)
    for e in feed.entries:
        yield {
            "title": e.get("title") or "",
            "url": e.get("link") or "",
            "summary": BeautifulSoup(e.get("summary",""), "html.parser").get_text()[:1000],
            "published": e.get("published") or e.get("updated") or "",
            "feed": url
        }

# ---- Reader ----
def extract_main(url, fallback=""):
    try:
        downloaded = trafilatura.fetch_url(url, timeout=12)
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
        text = normalize_text(text)
        if len(text) < 400:
            text = normalize_text(fallback)
        return text
    except Exception:
        return normalize_text(fallback)

def embed(texts):
    return MODEL.encode(texts, normalize_embeddings=True)

# ---- Intent scoring ----
def score_item(text, inc, exc, source_weight=0.0):
    e_item, e_inc, e_exc = embed([text, inc, exc])
    return float(util.cos_sim(e_item, e_inc) - util.cos_sim(e_item, e_exc) + source_weight)

# ---- Simple clustering ----
def cluster_items(items, sim_thr=0.72):
    texts = [f"{it['title']}. {it['text'][:2000]}" for it in items]
    if not texts: return []
    embs = embed(texts)
    sim = cosine_similarity(embs)
    clusters, used = [], set()
    n = len(items)
    for i in range(n):
        if i in used: continue
        group = [i]
        for j in range(i+1, n):
            if sim[i,j] >= sim_thr:
                group.append(j); used.add(j)
        clusters.append(group); used.add(i)
    return clusters

# ---- Summarization ----
def summarize(text, sentences=3):
    text = normalize_text(text)
    if not text: return ""
    parser = PlaintextParser.from_string(text, sumy_tokenize.Tokenizer("english"))
    summ = TextRankSummarizer()
    summ.stop_words = get_stop_words("english")
    sents = [str(s) for s in summ(parser.document, sentences)]
    return " ".join(sents)

# ---- Trends (NER + noun chunks) ----
import spacy
NLP = spacy.load("en_core_web_sm", disable=["lemmatizer","textcat","tagger"])
NLP.max_length = 2_000_000
SAFE_ENTS = {"ORG","PRODUCT","GPE","NORP","EVENT","WORK_OF_ART"}

def extract_terms(text):
    if not text: return []
    doc = NLP(text[:10000])
    ents = [e.text for e in doc.ents if e.label_ in SAFE_ENTS and 2 <= len(e.text) <= 60]
    chunks = [ch.text for ch in doc.noun_chunks if len(ch.text) >= 3]
    raw = ents + chunks
    terms = []
    for t in raw:
        tt = re.sub(r"[\s\-–—]+", " ", t).strip()
        tt = re.sub(r"[^\w\s&/\.]", "", tt).lower()
        if not tt or tt.isdigit() or len(tt) < 3: 
            continue
        tt = re.sub(r"\b(inc|ltd|plc|corp|co|company|group)\b\.?$", "", tt).strip()
        if tt: terms.append(tt)
    return terms

def update_trends(kept, hist, window_weeks, new_min_sources, momentum_jump_pct):
    today = datetime.utcnow().date().isoformat()
    term_counts = defaultdict(int)
    term_sources = defaultdict(set)
    for it in kept:
        tset = set(extract_terms(f"{it['title']}. {it['text']}"))
        for t in tset:
            term_counts[t] += 1
            term_sources[t].add(it["domain"])
    hist.setdefault("terms", {})
    emerging, momentum = [], []
    for t,c in term_counts.items():
        srcs = len(term_sources[t])
        series = hist["terms"].get(t, [])
        last = series[-1]["count"] if series else 0
        series.append({"date": today, "count": int(c), "sources": srcs})
        if len(series) > window_weeks: series = series[-window_weeks:]
        hist["terms"][t] = series
        prev_total = sum(x["count"] for x in series[:-1])
        if prev_total == 0 and srcs >= new_min_sources and c >= 2:
            emerging.append({"term": t, "count": int(c), "sources": srcs})
        elif last > 0:
            inc = ((c - last) / max(1, last)) * 100
            if inc >= momentum_jump_pct and c >= 2:
                momentum.append({"term": t, "count": int(c), "sources": srcs, "pct": round(inc)})
    emerging.sort(key=lambda x: (-x["count"], -x["sources"], x["term"]))
    momentum.sort(key=lambda x: (-x["pct"], -x["count"], x["term"]))
    return emerging[:10], momentum[:10], hist

# ---- Category bucketer & ranking/impact ----
def bucket(title, text):
    t = (title + " " + text[:1200]).lower()
    if any(w in t for w in CFG["ranking"]["launch_words"]): return "Product & Feature Signals"
    if any(w in t for w in CFG["ranking"]["funding_words"]): return "Strategic Moves"
    if any(w in t for w in CFG["ranking"]["reg_words"]):     return "Regulation & Risk"
    if any(n in t for n in ["mottola","jon younger","barry matthews","josh bersin","analyst","commentary","opinion"]):
        return "Influencer & Analyst Commentary"
    if any(w in t for w in ["ai","automation","apac","low-code","no-code","niche","entrepreneurs"]):
        return "Market & Trend Signals"
    return "Market & Trend Signals"

def estimate_impact(item, section):
    score = item["score"]
    url = item["url"].lower()
    text = (item["title"] + " " + item["text"][:800]).lower()
    w = CFG["scoring"]["weights"].get(section, 1.0)
    if any(d in url for d in CFG["ranking"]["big_vendor_domains"]): score += 0.08
    if any(wd in text for wd in CFG["ranking"]["funding_words"]): score += 0.07
    if any(wd in text for wd in CFG["ranking"]["contract_words"]): score += 0.06
    if any(wd in text for wd in CFG["ranking"]["reg_words"]):     score += 0.06
    if any(wd in text for wd in CFG["ranking"]["launch_words"]):  score += 0.05
    score += 0.02 * item.get("cluster_size", 1)
    s = score * w
    impact = "High" if s >= 0.55 else "Medium" if s >= 0.42 else "Low"
    return s, impact

def build_top10(kept, clusters):
    rows = []
    for g in clusters:
        group = [kept[i] for i in g]
        main = group[0]
        sec = bucket(main["title"], main["text"])
        main["cluster_size"] = len(group)
        s, imp = estimate_impact(main, sec)
        rows.append({
            "title": main["title"],
            "url": main["url"],
            "section": sec,
            "score": round(s, 3),
            "impact": imp,
            "summary_hint": group[0]["text"][:1400]
        })
    rows.sort(key=lambda r: (-r["score"], r["title"]))
    return rows[:CFG["ranking"]["top_k"]]

# ---- Report rendering ----
def render_report(today, bullets_by_section, top10, emerging, momentum):
    out = [f"# Weekly FMS Brief — {today}\n",
           "## TL;DR\n"]
    # TL;DR from top 6 items
    for r in top10[:6]:
        out.append(f"- **{r['title']}** — {r['impact']} impact. [Source]({r['url']})")

    if emerging or momentum:
        out.append("\n## Emerging terms\n")
        chips = []
        for e in emerging[:8]: chips.append(f"`{e['term']}` ({e['count']})")
        if chips: out.append("**New this week:** " + " • ".join(chips))
        chips = []
        for m in momentum[:6]: chips.append(f"`{m['term']}` (+{m['pct']}%)")
        if chips: out.append("**Momentum:** " + " • ".join(chips))

    out.append("\n## Top 10 News Items\n")
    for i, r in enumerate(top10, 1):
        out += [
            f"**{i}. {r['title']}**",
            f"{summarize(r['summary_hint'], sentences=2)}",
            f"**Impact:** {r['impact']} — [Source]({r['url']})",
            ""
        ]

    order = ["Product & Feature Signals","Strategic Moves","Market & Trend Signals","Regulation & Risk","Influencer & Analyst Commentary","Discovery Highlights"]
    for sec in order:
        if bullets_by_section.get(sec):
            out.append(f"\n## {sec}\n")
            out += [f"- {b}" for b in bullets_by_section[sec][:CFG["max_items_per_section"]]]
    return "\n".join(out)

# ---- Discovery bookkeeping ----
def add_discovery(feed_url, reason):
    DISC["pending"].setdefault(feed_url, {"weeks":0,"reason":reason})

def promote_discoveries():
    promoted = []
    for f, meta in list(DISC.get("pending", {}).items()):
        if meta.get("weeks",0) >= SRC["discovery"]["min_weeks_to_promote"]:
            DISC["feeds"][f] = {"added": datetime.utcnow().isoformat(), "reason": meta.get("reason","")}
            del DISC["pending"][f]
            promoted.append(f)
    return promoted

# ---- MAIN ----
def main():
    inc, exc = CFG["intent"]["include"], CFG["intent"]["exclude"]
    keep_thr = CFG["scoring"]["keep_threshold"]

    # 1) Build feed list
    feeds = set(SRC.get("rss", []))
    # existing discovered feeds
    feeds |= set(DISC.get("feeds", {}).keys())
    # resolve domains to feeds
    for dom in SRC.get("domains", []):
        new = discover_feeds_for_domain(dom)
        feeds |= set(new)

    # 2) Collect
    items = []
    for feed in sorted(feeds):
        try:
            for it in fetch_feed(feed):
                if not it["url"]: continue
                if not is_recent(it["published"], days=CFG["lookback_days"], tzname=CFG["timezone"]):
                    continue
                it["domain"] = domain(it["url"])
                items.append(it)
        except Exception:
            continue

    # 3) Read
    for it in items:
        it["text"] = extract_main(it["url"], fallback=it["summary"])
        it["title"] = normalize_text(it["title"])

    # 4) Score & keep
    kept = []
    for it in items:
        sw = float(HIST["sources"].get(it["domain"], 0.0))
        s = score_item(f"{it['title']}. {it['text']}", inc, exc, sw)
        it["score"] = s
        if s >= keep_thr and len(it["text"]) > 300:
            kept.append(it)

    # 5) Dedupe & Cluster
    # simple URL-based dedupe
    seen = set(); kept = [k for k in kept if not (k["url"] in seen or seen.add(k["url"]))]
    clusters = cluster_items(kept, sim_thr=0.72)

    # 6) Build bullets per section
    sections = defaultdict(list)
    for g in clusters:
        group = [kept[i] for i in g]
        blob = "\n\n".join([gi["text"] for gi in group])[:8000]
        main = group[0]
        sec = bucket(main["title"], blob)
        summary = summarize(blob, sentences=3)
        bullet = f"{summary} *(sources: " + ", ".join([f'[{"1" if i==0 else str(i+1)}]({gi["url"]})' for i,gi in enumerate(group[:5])]) + ")*"
        sections[sec].append(bullet)

    # 7) Top10 ranking/impact
    top10 = build_top10(kept, clusters)

    # 8) Trends
    emerging, momentum, new_hist = update_trends(
        kept=kept, hist=HIST,
        window_weeks=CFG["trends"]["window_weeks"],
        new_min_sources=CFG["trends"]["new_min_sources"],
        momentum_jump_pct=CFG["trends"]["momentum_jump_pct"],
    )
    HIST.update(new_hist)

    # 9) Domain reputation nudge
    for it in kept:
        d = it["domain"]
        HIST["sources"][d] = max(-0.05, min(0.10, HIST["sources"].get(d, 0.0) + 0.01))

    # 10) Discovery (from kept links)
    new_feeds = []
    if SRC.get("discovery", {}).get("expand_from_kept_links", True):
        cap = SRC["discovery"].get("max_new_sources_per_week", 3)
        for it in kept:
            try:
                res = requests_get(it["url"])
                if not res.ok: continue
                soup = BeautifulSoup(res.text, "html.parser")
                for link in soup.find_all("link", {"rel":"alternate"}):
                    if "rss" in (link.get("type") or "") or "atom" in (link.get("type") or ""):
                        href = link.get("href")
                        if href and href.startswith("http") and href not in feeds and href not in DISC.get("feeds", {}) and href not in DISC.get("pending", {}):
                            fp = feedparser.parse(href)
                            if fp.bozo == 0 and fp.entries:
                                add_discovery(href, reason=f"seen via {it['domain']}")
                                new_feeds.append(href)
                                if len(new_feeds) >= cap: break
                if len(new_feeds) >= cap: break
            except Exception:
                continue

    # advance pending counters
    for f in list(DISC.get("pending", {}).keys()):
        DISC["pending"][f]["weeks"] = DISC["pending"][f].get("weeks",0) + 1
    promoted = promote_discoveries()

    # 11) Render report
    today = datetime.utcnow().date().isoformat()
    report_md = render_report(today, sections, top10, emerging, momentum)
    (REPORT_DIR / f"{today}.md").write_text(report_md, encoding="utf-8")

    # 12) Save data snapshots
    HIST_PATH.write_text(json.dumps(HIST, indent=2))
    DISC_PATH.write_text(yaml.safe_dump(DISC, sort_keys=False))

    # CSV / JSON of kept items
    items_csv = REPORT_DIR / f"items-{today}.csv"
    items_json = REPORT_DIR / f"items-{today}.json"
    with items_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title","url","domain","score"])
        for it in kept:
            w.writerow([it["title"], it["url"], it["domain"], f"{it['score']:.3f}"])
    items_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2))

    # 13) Print quick summary for logs
    print(f"Kept {len(kept)} items across {len(clusters)} clusters. New feeds queued: {len(new_feeds)}. Promoted: {len(promoted)}.")

if __name__ == "__main__":
    main()
