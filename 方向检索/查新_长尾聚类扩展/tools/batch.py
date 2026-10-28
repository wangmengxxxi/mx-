# -*- coding: utf-8 -*-
"""Batch OpenAlex queries, write UTF-8 readable report.
Input: tools/queries.txt lines: tag || query || from_date || n || sort
"""
import sys, json, os, time, urllib.parse, urllib.request, re

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
RAW = os.path.join(BASE, "raw", "openalex")
os.makedirs(RAW, exist_ok=True)
MAIL = "research@example.org"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 + 2 * i)
    return {"__error__": str(last), "results": [], "meta": {}}


def slim(w):
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "id": w.get("id"), "doi": w.get("doi"),
        "title": w.get("title") or w.get("display_name") or "",
        "year": w.get("publication_year"), "date": w.get("publication_date"),
        "venue": src.get("display_name"), "type": w.get("type"),
        "cited": w.get("cited_by_count"),
        "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:6],
        "oa": (w.get("best_oa_location") or {}).get("pdf_url") or (w.get("best_oa_location") or {}).get("landing_page_url"),
    }


def run(tag, q, frm, n, sort="relevance"):
    params = {"per-page": str(min(int(n), 100)), "mailto": MAIL,
              "filter": "title_and_abstract.search:" + q}
    if frm and frm != "-":
        params["filter"] += ",from_publication_date:" + frm
    if sort == "cited":
        params["sort"] = "cited_by_count:desc"
    d = get("https://api.openalex.org/works?" + urllib.parse.urlencode(params))
    out = [slim(w) for w in d.get("results", [])]
    fn = os.path.join(RAW, re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)[:70] + ".json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "query": q, "from": frm, "total": d.get("meta", {}).get("count"), "results": out},
                  f, ensure_ascii=False, indent=1)
    return d.get("meta", {}).get("count"), out, d.get("__error__")


def main():
    qfile = sys.argv[1]
    outpath = sys.argv[2]
    lines = [l.strip() for l in open(qfile, encoding="utf-8") if l.strip() and not l.startswith("#")]
    buf = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("||")]
        tag, q = parts[0], parts[1]
        frm = parts[2] if len(parts) > 2 else "2022-01-01"
        n = parts[3] if len(parts) > 3 else "40"
        sort = parts[4] if len(parts) > 4 else "relevance"
        total, out, err = run(tag, q, frm, n, sort)
        buf.append("=" * 100)
        buf.append("TAG=%s | QUERY=%s | FROM=%s | TOTAL_MATCHED=%s | SHOWN=%d %s" % (tag, q, frm, total, len(out), ("ERR=" + err) if err else ""))
        for i, w in enumerate(out, 1):
            buf.append("%2d. [%s] %s\n     venue=%s | cit=%s | %s" % (i, w["year"], w["title"][:170], (w["venue"] or "-")[:60], w["cited"], w["doi"] or w["id"]))
        time.sleep(1)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
    print("WROTE", outpath, "queries=", len(lines))


if __name__ == "__main__":
    main()
