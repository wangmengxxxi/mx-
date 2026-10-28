# -*- coding: utf-8 -*-
"""Citation tracking: fetch works citing a root paper, filter titles by keywords.
Usage: python citedby.py <doi-or-openalex-id> "kw1;kw2;..." [maxpages]
Also prints root info and total citing count; saves json to raw/openalex/cite_<tag>.json
"""
import sys, json, os, time, urllib.parse, urllib.request, re

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
RAW = os.path.join(BASE, "raw", "openalex")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
MAIL = "research@example.org"


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 + 2 * i)
    return {"__error__": str(last), "results": []}


def slim(w):
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {"title": w.get("title") or w.get("display_name") or "", "year": w.get("publication_year"),
            "venue": src.get("display_name"), "doi": w.get("doi"), "cited": w.get("cited_by_count"),
            "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:4]}


def main():
    pid = sys.argv[1]
    kws = [k.strip().lower() for k in sys.argv[2].split(";") if k.strip()]
    pages = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if pid.startswith("http"):
        pid = pid.rstrip("/").split("/")[-1]
    if pid.upper().startswith("W"):
        root = get("https://api.openalex.org/works/%s?mailto=%s" % (pid, MAIL))
    else:
        root = get("https://api.openalex.org/works/https://doi.org/%s?mailto=%s" % (pid, MAIL))
    if "__error__" in root:
        print("ROOT ERR", root["__error__"])
        return
    rid = root["id"].split("/")[-1]
    print("ROOT: %s | %s | citing=%s | id=%s" % (root.get("title"), root.get("publication_year"), root.get("cited_by_count"), rid))
    allc = []
    for pg in range(1, pages + 1):
        u = ("https://api.openalex.org/works?filter=cites:%s&per-page=200&page=%d&sort=publication_date:desc&mailto=%s"
             % (rid, pg, MAIL))
        d = get(u)
        rs = d.get("results", [])
        if not rs:
            break
        allc += [slim(w) for w in rs]
        time.sleep(0.5)
    print("fetched citing works:", len(allc), "of", root.get("cited_by_count"))
    hits = [c for c in allc if any(k in (c["title"] or "").lower() for k in kws)]
    print("--- keyword-matched (%d) ---" % len(hits))
    for c in sorted(hits, key=lambda x: -(x["year"] or 0)):
        print("  [%s] %s\n      venue=%s | %s" % (c["year"], c["title"][:150], (c["venue"] or "-")[:50], c["doi"]))
    fn = os.path.join(RAW, "cite_" + re.sub(r"[^A-Za-z0-9]+", "_", (root.get("title") or rid))[:50] + ".json")
    json.dump({"root": slim(root) | {"id": rid}, "n_citing_fetched": len(allc), "matches": hits, "all": allc},
              open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", os.path.basename(fn))


if __name__ == "__main__":
    main()
