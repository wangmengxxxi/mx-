# -*- coding: utf-8 -*-
"""Pick papers of interest from saved openalex json by title substring; print metadata + oa urls."""
import sys, json, os, glob

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
RAW = os.path.join(BASE, "raw", "openalex")


def load_all():
    seen = {}
    for fn in glob.glob(os.path.join(RAW, "*.json")):
        if os.path.basename(fn).startswith("cite_"):
            continue
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        for w in d.get("results", []):
            k = (w.get("title") or "").lower().strip()
            if k and k not in seen:
                seen[k] = w
    return seen


def main():
    keys = [k.lower() for k in sys.argv[1:]]
    seen = load_all()
    for k in keys:
        hits = [(t, w) for t, w in seen.items() if k in t]
        if not hits:
            print("### NO MATCH:", k)
            continue
        for t, w in hits:
            print("### %s" % w["title"])
            print("    year=%s | venue=%s | doi=%s | cit=%s" % (w["year"], w["venue"], w["doi"], w["cited"]))
            print("    authors=%s" % ", ".join(w.get("authors") or []))
            print("    oa=%s" % w.get("oa"))
            print("    id=%s" % w.get("id"))


if __name__ == "__main__":
    main()
