"""Strict full-title + full-author verification of every arXiv entry."""
import re, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time

NS = {'a': 'http://www.w3.org/2005/Atom'}

def parse_bib(path):
    text = open(path, encoding='utf-8').read()
    entries, i = [], 0
    while True:
        m = re.search(r'@(\w+)\s*\{\s*([^,]+),', text[i:])
        if not m:
            break
        kind, key = m.group(1), m.group(2).strip()
        start = i + m.end()
        depth, j = 1, start
        while j < len(text) and depth > 0:
            depth += 1 if text[j] == '{' else (-1 if text[j] == '}' else 0)
            j += 1
        block = text[start:j - 1]
        def field(nm):
            fm = re.search(nm + r'\s*=\s*\{', block)
            if not fm:
                return ''
            s, d, k = fm.end(), 1, fm.end()
            while k < len(block) and d > 0:
                d += 1 if block[k] == '{' else (-1 if block[k] == '}' else 0)
                k += 1
            return block[s:k - 1]
        entries.append((key, field('title'), field('author'), field('eprint')))
        i = start
    return entries

def norm(s):
    s = re.sub(r'\\[a-zA-Z]+\s*', '', s)
    s = re.sub(r'[{}$\\]', '', s)
    s = re.sub(r'\s+', '', s)
    return re.sub(r'[^a-z0-9]', '', s.lower())

entries = parse_bib('paper/custom.bib')
arx = [(k, t, a, e) for (k, t, a, e) in entries if e]
print('%d arXiv entries; strict title + author check' % len(arx))
for k in range(0, len(arx), 5):
    batch = [e[3] for e in arx[k:k + 5]]
    url = 'https://export.arxiv.org/api/query?id_list=' + ','.join(batch)
    with urllib.request.urlopen(url, timeout=40) as r:
        data = r.read().decode('utf-8', 'ignore')
    root = ET.fromstring(data)
    live = {}
    for e in root.findall('a:entry', NS):
        aid = e.findtext('a:id', default='', namespaces=NS).split('/abs/')[-1].split('v')[0]
        lt = ' '.join(e.findtext('a:title', default='', namespaces=NS).split())
        la = [a.findtext('a:name', default='', namespaces=NS) for a in e.findall('a:author', NS)]
        live[aid] = (lt, la)
    for (key, t, a, aid) in arx[k:k + 5]:
        lt, la = live.get(aid, ('', []))
        tmatch = norm(t) == norm(lt)
        # author match: compare last names of bib author field vs live list
        bib_lasts = [x.strip().split(',')[0] for x in a.split(' and ')]
        bib_lasts = [x.split()[-1] for x in bib_lasts]
        live_lasts = [x.split()[-1] for x in la]
        amatch = sorted(norm(x) for x in bib_lasts) == sorted(norm(x) for x in live_lasts)
        print('%-20s %-11s %s %s' % (key, aid, 'T' if tmatch else 't-DIFF', 'A' if amatch else 'a-DIFF'))
        if not tmatch:
            print('   bib  :', t[:110])
            print('   live :', lt[:110])
        if not amatch:
            print('   bibA :', a[:110])
            print('   liveA:', ', '.join(la)[:110])
    time.sleep(3)
