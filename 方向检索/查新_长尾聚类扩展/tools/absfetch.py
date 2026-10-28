# -*- coding: utf-8 -*-
"""Fetch OpenAlex work metadata + reconstructed abstract by DOI/OpenAlex id.
Usage: python absfetch.py <doi-or-id> [<doi-or-id> ...]
Writes to raw/abstracts.txt (append) and stdout."""
import sys, json, os, time, urllib.parse, urllib.request

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
OUT = os.path.join(BASE, "raw", "abstracts.txt")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


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
    return {"__error__": str(last)}


def recon(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[k] for k in sorted(pos))


def one(pid):
    pid = pid.strip()
    if pid.startswith("http"):
        pid = pid.rstrip("/").split("/")[-1]
    d = get("https://api.openalex.org/works/https://doi.org/%s?mailto=research@example.org" % pid
            if not pid.upper().startswith("W") else
            "https://api.openalex.org/works/%s?mailto=research@example.org" % pid)
    if "__error__" in d:
        return "ERR %s :: %s" % (pid, d["__error__"])
    src = ((d.get("primary_location") or {}).get("source") or {}).get("display_name")
    lines = []
    lines.append("=" * 90)
    lines.append("TITLE: " + (d.get("title") or ""))
    lines.append("YEAR: %s | VENUE: %s | TYPE: %s | CITED: %s" % (d.get("publication_year"), src, d.get("type"), d.get("cited_by_count")))
    lines.append("AUTHORS: " + ", ".join(a["author"]["display_name"] for a in (d.get("authorships") or [])))
    lines.append("DOI: " + str(d.get("doi")))
    b = d.get("best_oa_location") or {}
    lines.append("OA: " + str(b.get("pdf_url") or b.get("landing_page_url")))
    lines.append("ABSTRACT: " + (recon(d.get("abstract_inverted_index")) or "(none)"))
    return "\n".join(lines)


if __name__ == "__main__":
    buf = []
    for pid in sys.argv[1:]:
        t = one(pid)
        print(t)
        buf.append(t)
        time.sleep(0.6)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
