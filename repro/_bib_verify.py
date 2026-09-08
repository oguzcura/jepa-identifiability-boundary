"""Integrity audit: verify every arXiv eprint in custom.bib against the live
arXiv API (id_list + title comparison). Robust brace-counting parser.

Usage: python _bib_verify.py paper/custom.bib
"""
import sys, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import time

NS = {'a': 'http://www.w3.org/2005/Atom'}


def parse_bib(path):
    """Split bib into entries; for each: key, title, eprint (arXiv only)."""
    text = open(path, encoding='utf-8').read()
    entries = []
    i = 0
    while True:
        m = re.search(r'@(\w+)\s*\{\s*([^,]+),', text[i:])
        if not m:
            break
        kind, key = m.group(1), m.group(2).strip()
        start = i + m.end()
        # brace-count to end of entry
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        block = text[start:j - 1]
        def field(name):
            fm = re.search(name + r'\s*=\s*\{', block)
            if not fm:
                return ''
            s = fm.end()
            d, k = 1, s
            while k < len(block) and d > 0:
                if block[k] == '{':
                    d += 1
                elif block[k] == '}':
                    d -= 1
                k += 1
            return block[s:k - 1]
        title, eprint = field('title'), field('eprint')
        arch = field('archivePrefix')
        if eprint and 'arXiv' in arch:
            entries.append((key, kind, title, eprint.strip()))
        i = start
    return entries


def fetch_titles(aids):
    """Batch id_list query; return {aid: title}."""
    out = {}
    for k in range(0, len(aids), 10):
        batch = aids[k:k + 10]
        url = 'https://export.arxiv.org/api/query?id_list=' + ','.join(batch)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read().decode('utf-8', 'ignore')
                root = ET.fromstring(data)
                for e in root.findall('a:entry', NS):
                    aid = (e.findtext('a:id', default='', namespaces=NS)
                            .replace('http://arxiv.org/abs/', '')
                            .replace('https://arxiv.org/abs/', ''))
                    aid = re.sub(r'v\d+$', '', aid)  # drop version suffix
                    t = e.findtext('a:title', default='', namespaces=NS)
                    out[aid] = ' '.join(t.split())
                break
            except Exception as ex:
                if attempt == 2:
                    print('  ERR fetching batch %s: %s' % (batch, ex))
                time.sleep(2)
        time.sleep(3)
    return out


def norm(s):
    s = re.sub(r'\\[a-zA-Z]+\s*', '', s)
    s = re.sub(r'[{}$]', '', s)
    return re.sub(r'[^a-z0-9]', '', s.lower())


def main():
    entries = parse_bib(sys.argv[1])
    print('parsed %d arXiv entries' % len(entries))
    aids = [e[3] for e in entries]
    live = fetch_titles(list(set(aids)))
    print('live titles fetched: %d/%d' % (len(live), len(set(aids))))
    bad = 0
    for key, kind, title, aid in entries:
        real = live.get(aid, '')
        if not real:
            print('%-22s %-12s MISSING-from-API' % (aid, key)); bad += 1
            continue
        c, r = norm(title), norm(real)
        ok = (c[:20] in r) or (r[:20] in c) or (c[:10] in r)
        tag = 'OK ' if ok else 'BAD'
        if not ok:
            bad += 1
        print('%-22s %-12s %s | bib="%.48s" | real="%.48s"' % (aid, key, tag, title, real))
    print('\n=== %d entries, %d mismatches ===' % (len(entries), bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
