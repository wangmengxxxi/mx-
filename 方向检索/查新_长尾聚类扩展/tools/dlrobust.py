# -*- coding: utf-8 -*-
"""Robust PDF downloader with HTTP Range resume + EOF verification."""
import os, sys, time, subprocess

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
OUT = os.path.join(BASE, "raw", "pdfs")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def has_eof(p):
    try:
        n = os.path.getsize(p)
        if n < 1000:
            return False
        with open(p, "rb") as f:
            f.seek(max(0, n - 2048))
            return b"%%EOF" in f.read()
    except Exception:
        return False


def fetch(name, url, max_rounds=12):
    path = os.path.join(OUT, name)
    if has_eof(path):
        print("SKIP", name, os.path.getsize(path))
        return
    for r in range(max_rounds):
        before = os.path.getsize(path) if os.path.exists(path) else 0
        cmd = ["curl.exe", "-sL", "-C", "-", "--max-time", "180", "-A", UA, url, "-o", path]
        subprocess.run(cmd)
        after = os.path.getsize(path) if os.path.exists(path) else 0
        ok = has_eof(path)
        print("  round%d %s %d -> %d eof=%s" % (r + 1, name, before, after, ok), flush=True)
        if ok:
            return
        if after <= before:
            time.sleep(3)
    print("GIVEUP", name, os.path.getsize(path) if os.path.exists(path) else 0)


if __name__ == "__main__":
    jobs = []
    for ln in open(sys.argv[1], encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        n, u = [x.strip() for x in ln.split("||")]
        jobs.append((n, u))
    for n, u in jobs:
        fetch(n, u)
    print("ALL DONE")
