# -*- coding: utf-8 -*-
"""Report source lines whose ASCII double-quote count is not a multiple of 2 (likely nested quotes), plus lines with 4+ quotes inside a single field."""
import io, re

base = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
p = base + r"\tools\mkcsv.py"
lines = io.open(p, encoding="utf-8").read().split("\n")
for i, ln in enumerate(lines, 1):
    if 'a("' not in ln and '  "' not in ln:
        continue
    # heuristic: count quotes; a well-formed line has even count, but nested quotes also give even.
    # detect pattern: a quote that is followed by a non-ASCII char but preceded by ASCII non-quote, or vice versa
    for m in re.finditer(r'"', ln):
        j = m.start()
        prev = ln[j - 1] if j > 0 else ""
        nxt = ln[j + 1] if j + 1 < len(ln) else ""
        if (prev and ord(prev) > 127 and nxt and ord(nxt) < 128 and nxt not in ',")') or \
           (prev and ord(prev) < 128 and prev not in '(:, "' and nxt and ord(nxt) > 127):
            print(i, "::", ln[:200])
            break
