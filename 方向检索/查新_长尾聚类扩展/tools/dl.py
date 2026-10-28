# -*- coding: utf-8 -*-
"""Download URLs to raw/pdfs with polite delay. Input file: lines "filename || url" """
import sys, os, time, urllib.request, re

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
OUT = os.path.join(BASE, "raw", "pdfs")
os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def dl(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data


def main():
    lst = sys.argv[1]
    for ln in open(lst, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name, url = [p.strip() for p in ln.split("||")]
        path = os.path.join(OUT, name)
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            print("SKIP", name)
            continue
        try:
            d = dl(url, path)
            ok = d[:4] == b"%PDF"
            print("OK" if ok else "WARN-notpdf", name, len(d), "hdr=", d[:5])
        except Exception as e:
            print("FAIL", name, url, str(e)[:120])
        time.sleep(1.5)


if __name__ == "__main__":
    main()
