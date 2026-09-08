"""Fetch live arXiv metadata (exact title + authors) for the anonymous/paraphrased entries."""
import urllib.request, xml.etree.ElementTree as ET, time

NS = {'a': 'http://www.w3.org/2005/Atom'}

ids = {
    'mcjepa2026': '2608.13621',
    'nogaussian2026': '2608.17542',
    'jepaparadox2026': '2607.23531',
    'phylatent2026': '2608.05720',
    'discretejepa2025': '2506.14373',
    'sjepa2026': '2608.04060',
    'rijepa2026': '2603.13265',
    'aimprobing2026': '2603.20327',
    'controlledWM2026': '2607.22430',
    'causaljepa2026': '2602.11389',
    'uwmjepa2026b': '2605.25313',
    'dsge2026': '2607.03144',
}
aids = list(ids.values())
for k in range(0, len(aids), 5):
    batch = aids[k:k + 5]
    url = 'https://export.arxiv.org/api/query?id_list=' + ','.join(batch)
    with urllib.request.urlopen(url, timeout=40) as r:
        data = r.read().decode('utf-8', 'ignore')
    root = ET.fromstring(data)
    for e in root.findall('a:entry', NS):
        aid = e.findtext('a:id', default='', namespaces=NS).split('/abs/')[-1]
        aid = aid.split('v')[0]
        title = ' '.join(e.findtext('a:title', default='', namespaces=NS).split())
        auths = [a.findtext('a:name', default='', namespaces=NS) for a in e.findall('a:author', NS)]
        pub = e.findtext('a:published', default='', namespaces=NS)
        print('###', aid)
        print('TITLE:', title)
        print('AUTHORS:', ' and '.join(auths))
        print('PUBLISHED:', pub[:10])
        print('---')
    time.sleep(3)
