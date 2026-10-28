# -*- coding: utf-8 -*-
"""OpenReview (api2) search helper. Usage: python orsearch.py "query" [limit]
Dumps compact results + saves json. Also supports `pdf <id> <outfile>`."""
import sys, json, os, urllib.parse, urllib.request, time

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
RAW = os.path.join(BASE, "raw", "openreview")
os.makedirs(RAW, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def get(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 + 2 * i)
    return {"__error__": str(last), "notes": []}


def val(n, k):
    v = (n.get("content") or {}).get(k)
    if isinstance(v, dict):
        return v.get("value")
    return v


def main():
    cmd = sys.argv[1]
    if cmd == "search":
        q = sys.argv[2]
        lim = sys.argv[3] if len(sys.argv) > 3 else "20"
        d = get("https://api2.openreview.net/notes/search?term=%s&limit=%s&source=forum" % (urllib.parse.quote(q), lim))
        notes = d.get("notes", [])
        out = []
        print("QUERY:", q, "| returned:", len(notes), "| err:", d.get("__error__", ""))
        for n in notes:
            t = val(n, "title")
            v = val(n, "venue")
            if isinstance(v, list):
                v = " / ".join(str(x) for x in v)
            au = val(n, "authors")
            if isinstance(au, list):
                au = ", ".join([(a if isinstance(a, str) else str(a.get("fullname") or a.get("username") or a)) for a in au[:5]])
            item = {"id": n.get("id"), "forum": n.get("forum"), "title": t, "venue": v,
                    "authors": au, "abstract": val(n, "abstract"), "keywords": val(n, "keywords"),
                    "year": time.strftime("%Y", time.localtime((n.get("cdate") or 0) / 1000))}
            out.append(item)
            print(" - [%s] %s\n     venue=%s | id=%s | %s" % (item["year"], t, v, item["id"], au))
        fn = os.path.join(RAW, "or_" + "".join(c if c.isalnum() else "_" for c in q)[:60] + ".json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("saved:", os.path.basename(fn))
    elif cmd == "pdf":
        nid, out = sys.argv[2], sys.argv[3]
        req = urllib.request.Request("https://openreview.net/pdf?id=" + nid, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        with open(out, "wb") as f:
            f.write(data)
        print("saved", out, len(data), "bytes, header=", data[:5])


if __name__ == "__main__":
    main()
