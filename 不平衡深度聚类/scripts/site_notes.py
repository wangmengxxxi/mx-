# -*- coding: utf-8 -*-
import json
import re
import sys
import html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
p=ROOT/'results/papernotes_search_index_full.json'
if not p.exists():
    p=ROOT/'results/papernotes_search_index.json'
s=p.read_text(encoding='utf-8',errors='replace')
try:
    data=json.loads(s)
    docs=data['docs']
    complete=True
except json.JSONDecodeError:
    docs=[]
    decoder=json.JSONDecoder()
    for m in re.finditer(r'\{"location":',s):
        try:
            d,end=decoder.raw_decode(s,m.start())
            docs.append(d)
        except json.JSONDecodeError:
            pass
    complete=False
hits=[d for d in docs if re.search(r'imbalanc|long.tail|P.?OT|cluster|聚类|长尾|不平衡',d.get('title',''),re.I)]
(ROOT/'results/papernotes_matches.json').write_text(json.dumps({'complete_index':complete,'indexed_records':len(docs),'matches':hits},ensure_ascii=False,indent=2),encoding='utf-8')
print('index_complete',complete,'records',len(docs),'matches',len(hits))
for d in hits:
    if re.search(r'imbalanc.*cluster|cluster.*imbalanc|long.tail.*cluster|cluster.*long.tail|\bP.{0,2}OT\b|deep.*cluster|image.*cluster|distribution.agnostic',d['title'],re.I):
        print(d['title'],'https://papernotes.org/'+d['location'],d.get('text',''))
if '--read-pages' in sys.argv:
    for p in [ROOT/'results'/f'papernotes_{name}.html' for name in ['minicluster','samples','cae']]:
        text=p.read_text(encoding='utf-8')
        article=text[text.find('<article'):text.find('</article>')]
        plain=html.unescape(re.sub(r'<[^>]+>',' ',article))
        print('\nFILE',p.name,'\n',re.sub(r'\n\s*\n','\n',plain))
        print('SOURCE LINKS',re.findall(r'href="(https?://(?:arxiv|openreview|github|openaccess|ojs)[^"]+)"',article))
