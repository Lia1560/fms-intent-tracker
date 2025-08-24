def main():
    inc, exc = CFG["intent"]["include"], CFG["intent"]["exclude"]
    keep_thr = CFG["scoring"]["keep_threshold"]

    print(">>> Step 1: Building feed list")
    feeds = set(SRC.get("rss", []))
    feeds |= set(DISC.get("feeds", {}).keys())
    for dom in SRC.get("domains", []):
        new = discover_feeds_for_domain(dom)
        if new:
            print(f"  discovered {len(new)} feeds for {dom}")
        feeds |= set(new)
    print(f"Total feeds to check: {len(feeds)}")

    print(">>> Step 2: Collecting feed items")
    items = []
    for feed in sorted(feeds):
        try:
            for it in fetch_feed(feed):
                if not it["url"]:
                    continue
                if not is_recent(it["published"], days=CFG["lookback_days"], tzname=CFG["timezone"]):
                    continue
                it["domain"] = domain(it["url"])
                items.append(it)
        except Exception as e:
            print(f"  feed failed: {feed} ({e})")
            continue
    print(f"Collected {len(items)} raw items")

    print(">>> Step 3: Fetching article text")
    for it in items:
        it["text"] = extract_main(it["url"], fallback=it["summary"])
        it["title"] = normalize_text(it["title"])
    print("Fetched article text for all items")

    print(">>> Step 4: Scoring items")
    kept = []
    for it in items:
        sw = float(HIST["sources"].get(it["domain"], 0.0))
        s = score_item(f"{it['title']}. {it['text']}", inc, exc, sw)
        it["score"] = s
        if s >= keep_thr and len(it["text"]) > 300:
            kept.append(it)
    print(f"Kept {len(kept)} items")

    print(">>> Step 5: Clustering kept items")
    seen = set()
    kept = [k for k in kept if not (k["url"] in seen or seen.add(k["url"]))]
    clusters = cluster_items(kept, sim_thr=0.72)
    print(f"Formed {len(clusters)} clusters")

    print(">>> Step 6: Building bullets per section")
    sections = defaultdict(list)
    for g in clusters:
        group = [kept[i] for i in g]
        blob = "\n\n".join([gi["text"] for gi in group])[:8000]
        main_item = group[0]
        sec = bucket(main_item["title"], blob)
        summary = summarize(blob, sentences=3)
        bullet = f"{summary} *(sources: " + ", ".join([f'[{"1" if i==0 else str(i+1)}]({gi["url"]})' for i,gi in enumerate(group[:5])]) + ")*"
        sections[sec].append(bullet)
    print("Section bullets built")

    print(">>> Step 7: Ranking Top 10")
    top10 = build_top10(kept, clusters)
    print(f"Top10 items: {len(top10)}")

    print(">>> Step 8: Updating trends")
    emerging, momentum, new_hist = update_trends(
        kept=kept, hist=HIST,
        window_weeks=CFG["trends"]["window_weeks"],
        new_min_sources=CFG["trends"]["new_min_sources"],
        momentum_jump_pct=CFG["trends"]["momentum_jump_pct"],
    )
    HIST.update(new_hist)
    print(f"Emerging: {len(emerging)}, Momentum: {len(momentum)}")

    print(">>> Step 9: Updating domain reputation")
    for it in kept:
        d = it["domain"]
        HIST["sources"][d] = max(-0.05, min(0.10, HIST["sources"].get(d, 0.0) + 0.01))

    print(">>> Step 10: Discovery from kept links")
    new_feeds = []
    if SRC.get("discovery", {}).get("expand_from_kept_links", True):
        cap = SRC["discovery"].get("max_new_sources_per_week", 3)
        for it in kept:
            try:
                res = requests_get(it["url"])
                if not res.ok: 
                    continue
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
            except Exception as e:
                print(f"  discovery failed for {it['url']}: {e}")
                continue
    print(f"New feeds queued: {len(new_feeds)}")

    print(">>> Step 11: Rendering report")
    today = datetime.utcnow().date().isoformat()
    report_md = render_report(today, sections, top10, emerging, momentum)

    if not report_md.strip():
        report_md = f"# Weekly FMS Brief — {today}\n\n_No items were collected this week._"

    outpath = REPORT_DIR / f"{today}.md"
    outpath.write_text(report_md, encoding="utf-8")
    print(f"Wrote report to {outpath}")

    print(">>> Step 12: Saving data snapshots")
    HIST_PATH.write_text(json.dumps(HIST, indent=2))
    DISC_PATH.write_text(yaml.safe_dump(DISC, sort_keys=False))

    items_csv = REPORT_DIR / f"items-{today}.csv"
    items_json = REPORT_DIR / f"items-{today}.json"
    with items_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title","url","domain","score"])
        for it in kept:
            w.writerow([it["title"], it["url"], it["domain"], f"{it['score']:.3f}"])
    items_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2))

    print(">>> Step 13: Done")
    print(f"Kept {len(kept)} items across {len(clusters)} clusters. New feeds queued: {len(new_feeds)}. Promoted: {len(promote_discoveries())}.")
