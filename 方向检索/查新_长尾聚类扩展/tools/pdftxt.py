# -*- coding: utf-8 -*-
"""Extract text from PDFs in raw/pdfs into raw/text/*.txt (page markers kept)."""
import sys, os, glob
from pypdf import PdfReader

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
PDF = os.path.join(BASE, "raw", "pdfs")
TXT = os.path.join(BASE, "raw", "text")
os.makedirs(TXT, exist_ok=True)

targets = sys.argv[1:] if len(sys.argv) > 1 else [os.path.basename(p) for p in glob.glob(os.path.join(PDF, "*.pdf"))]
for name in targets:
    p = os.path.join(PDF, name)
    if not os.path.exists(p):
        print("MISSING", name)
        continue
    try:
        r = PdfReader(p)
        out = []
        for i, pg in enumerate(r.pages, 1):
            out.append("\n===== PAGE %d =====\n" % i)
            out.append(pg.extract_text() or "")
        fn = os.path.join(TXT, os.path.splitext(name)[0] + ".txt")
        open(fn, "w", encoding="utf-8").write("".join(out))
        print("OK", name, "pages=", len(r.pages), "chars=", sum(len(x) for x in out))
    except Exception as e:
        print("FAIL", name, str(e)[:120])
