# -*- coding: utf-8 -*-
"""Replace ASCII double quotes that are surrounded by CJK chars with curly quotes."""
import io

base = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
p = base + r"\tools\mkcsv.py"
s = io.open(p, encoding="utf-8").read()


def is_cjk(ch):
    return bool(ch) and ("\u4e00" <= ch <= "\u9fff" or ch in "（）、，。；：")


out = []
i = 0
toggle = True
while i < len(s):
    ch = s[i]
    if ch == '"':
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if is_cjk(prev) and is_cjk(nxt):
            out.append("\u201c" if toggle else "\u201d")
            toggle = not toggle
            i += 1
            continue
    out.append(ch)
    i += 1
io.open(p, "w", encoding="utf-8").write("".join(out))
print("patched, replaced pairs:", not toggle)
