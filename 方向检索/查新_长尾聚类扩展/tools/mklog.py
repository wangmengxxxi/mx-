# -*- coding: utf-8 -*-
"""Parse BATCH_*.txt into markdown rows for search_log.md"""
import re, os, glob

BASE = r"F:\研究生毕设\方向检索\查新_长尾聚类扩展"
OUT = os.path.join(BASE, "raw", "openalex")

rows = []
for fn in sorted(glob.glob(os.path.join(OUT, "BATCH_*.txt"))):
    for ln in open(fn, encoding="utf-8"):
        m = re.match(r"TAG=(\S+) \| QUERY=(.*?) \| FROM=(\S+) \| TOTAL_MATCHED=(\S+) \| SHOWN=(\d+)", ln.strip())
        if m:
            tag, q, frm, total, shown = m.groups()
            rows.append((tag, q, frm, total, shown, os.path.basename(fn)))

lines = ["| # | TAG | 查询式（OpenAlex title_and_abstract.search） | 起始年 | 命中总数 | 已取回 | 批次文件 |",
         "|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    lines.append("| %d | %s | `%s` | %s | %s | %s | %s |" % (i, r[0], r[1], r[2], r[3], r[4], r[5]))
open(os.path.join(BASE, "raw", "openalex_queries_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("total queries:", len(rows))
