# -*- coding: utf-8 -*-
"""OpenAlex helper: search works, dump JSON to raw/openalex, print compact table.
Usage:
  python oa.py search "<query>" [--from 2022-01-01] [--n 50] [--tag tagname] [--sort relevance|cited]
  python oa.py cite <openalex_id_or_doi>   # citations (cited_by) + references
"""
import sys, json, os, time, urllib.parse, urllib.request, re

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
RAW = os.path.join(BASE, "raw", "openalex")
os.makedirs(RAW, exist_ok=True)
MAIL = "research@example.org"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 + 2 * i)
    raise RuntimeError("GET failed: %s :: %s" % (url, last))


def slim(w):
    return {
        "id": w.get("id"),
        "doi": w.get("doi"),
        "title": w.get("title") or (w.get("display_name") or ""),
        "year": w.get("publication_year"),
        "date": w.get("publication_date"),
        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "type": w.get("type"),
        "cited_by": w.get("cited_by_count"),
        "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:8],
        "oa_url": (w.get("best_oa_location") or {}).get("pdf_url")
                  or (w.get("best_oa_location") or {}).get("landing_page_url"),
        "topic": ((w.get("primary_topic") or {}).get("display_name")),
    }


def do_search(q, frm="2022-01-01", n=50, tag="q", sort="relevance", mode="ti"):
    params = {
        "per-page": str(min(int(n), 100)),
        "mailto": MAIL,
    }
    filt = []
    if mode == "ti":
        filt.append("title_and_abstract.search:" + q)
        params["filter"] = ",".join(filt)
    else:
        params["search"] = q
        params["filter"] = ""
    if frm:
        params["filter"] = (params["filter"] + "," if params["filter"] else "") + "from_publication_date:" + frm
    if sort == "cited":
        params["sort"] = "cited_by_count:desc"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    d = get(url)
    out = [slim(w) for w in d.get("results", [])]
    fn = os.path.join(RAW, re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)[:80] + ".json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"query": q, "count": d.get("meta", {}).get("count"), "results": out}, f, ensure_ascii=False, indent=1)
    print("QUERY: %s | from=%s | total_matched=%s | shown=%d | file=%s" % (q, frm, d.get("meta", {}).get("count"), len(out), os.path.basename(fn)))
    for i, w in enumerate(out, 1):
        print("%2d. [%s] %s || %s || cit=%s || %s" % (i, w["year"], w["title"][:150], (w["venue"] or "-")[:45], w["cited_by"], (w["doi"] or w["id"])))
    return out


def do_cite(pid):
    pid = pid.replace("https://openalex.org/", "")
    url = "https://api.openalex.org/works/%s?mailto=%s" % (urllib.parse.quote(pid), MAIL)
    w = get(url)
    print("### ROOT:", w.get("title"), "|", w.get("publication_year"), "|", (w.get("primary_location") or {}).get("source", {}).get("display_name") if (w.get("primary_location") or {}).get("source") else "-", "| cited_by_count=", w.get("cited_by_count"))
    refs = w.get("referenced_works") or []
    print("--- references: %d" % len(refs))
    # batch fetch references in chunks of 50
    for i in range(0, len(refs), 50):
        chunk = refs[i:i + 50]
        u = "https://api.openalex.org/works?filter=openalex_id:%s&per-page=50&mailto=%s" % ("|".join(c.replace("https://openalex.org/", "") for c in chunk), MAIL)
        d = get(u)
        for r in d.get("results", []):
            s = slim(r)
            print("  R [%s] %s || %s" % (s["year"], (s["title"] or "")[:130], (s["venue"] or "-")[:40]))
    print("--- cited by (openalex count=%s)" % w.get("cited_by_count"))
    u = "https://api.openalex.org/works?filter=cites:%s&per-page=100&sort=publication_date:desc&mailto=%s" % (pid, MAIL)
    d = get(u)
    print("  returned %d" % len(d.get("results", [])))
    for r in d.get("results", []):
        s = slim(r)
        print("  C [%s] %s || %s || %s" % (s["year"], (s["title"] or "")[:130], (s["venue"] or "-")[:35], s["doi"] or s["id"]))
    fn = os.path.join(RAW, "cite_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", pid)[-40:] + ".json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"root": slim(w), "n_refs": len(refs), "cited_by": [slim(r) for r in d.get("results", [])]}, f, ensure_ascii=False, indent=1)
    print("saved:", os.path.basename(fn))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        q = sys.argv[2]
        args = sys.argv[3:]
        kw = {"frm": "2022-01-01", "n": "50", "tag": "q", "sort": "relevance"}
        i = 0
        while i < len(args):
            if args[i] == "--from":
                kw["frm"] = args[i + 1]; i += 2
            elif args[i] == "--n":
                kw["n"] = args[i + 1]; i += 2
            elif args[i] == "--tag":
                kw["tag"] = args[i + 1]; i += 2
            elif args[i] == "--sort":
                kw["sort"] = args[i + 1]; i += 2
            elif args[i] == "--all":
                kw["frm"] = ""; i += 1
            else:
                i += 1
        do_search(q, **kw)
    elif cmd == "cite":
        do_cite(sys.argv[2])
