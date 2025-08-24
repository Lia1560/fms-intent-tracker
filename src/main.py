import os, json, yaml, csv, re, time, socket
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
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words

from dateutil import parser as dtparser, tz
import spacy

print(">>> src.main loaded")   # DEBUG: confirm module loaded

# Global socket timeout
socket.setdefaulttimeout(int(os.getenv("HTTP_TIMEOUT", "20")))

# ---- Paths & Config ----
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SRC = yaml.safe_load((ROOT / "sources.yaml").read_text())
HIST_PATH = ROOT / "data" / "history.json"
DISC_PATH = ROOT / "data" / "discovered_sources.yaml"
REPORT_DIR = ROOT / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(parents=True, exist_ok=True)

HIST = json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else {"terms":{}, "sources":{}}
DISC = yaml.safe_load(DISC_PATH.read_text()) if DISC_PATH.exists() else {"feeds":{}, "pending":{}}

# ---- Embeddings model ----
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Basic helpers ----
def normalize_text(s): return re.sub(r"\s+", " ", (s or "")).strip()

def domain(u):
    try:
        ext = tldextract.extract(u)
        return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except:
        return urlparse(u).netloc

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

# ---- Finder ----
COMMON_FEED_PATHS = ["/feed", "/rss", "/rss.xml", "/atom.xml", "/news/rss", "/blog/rss", "/press/rss", "/changelog.xml"]

def discover_feeds_for_domain(dom):
    found = set()
    try:
        for p in COMMON_FEED_PATHS:
            try:
                url = f"https://{dom}{p}"
                fp = feedparser.parse(url)
                if fp.bozo == 0 and fp.entries:
                    found.add(url)
            except Exception:
                pass
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
    return list(found)[:3]

# ---- Feed fetch ----
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

# ---- Clustering ----
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
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = TextRankSummarizer()
    summ.stop_words = get_stop_words("english")
    sents = [str(s) for s in summ(parser.document, sentences)]
    return " ".join(sents)

# ---- Trends ----
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

# ---- MAIN ----
def main():
    print(">>> Entered main()")   # DEBUG: confirm we reach main
    inc, exc = CFG["intent"]["include"], CFG["intent"]["exclude"]
    keep_thr = CFG["scoring"]["keep_threshold"]

    print(">>> Step 1: Building feed list")
    # ... rest of your existing main() body ...
    # (unchanged, keep all your other debug prints and logic)
    

if __name__ == "__main__":
    print(">>> __main__ guard hit")   # DEBUG: confirm execution path
    main()
