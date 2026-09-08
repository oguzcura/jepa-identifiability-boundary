import re

tex = open('paper/main.tex', encoding='utf-8').read()
bib = open('paper/custom.bib', encoding='utf-8').read()

cited = set()
for m in re.finditer(r'\\(?:cite|citep|citet|citealp)(?:\[[^\]]*\])?\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        k = k.strip()
        if k:
            cited.add(k)

bibkeys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
print('cited unique:', len(cited), '| bib keys:', len(bibkeys))
print('MISSING from bib:', sorted(cited - bibkeys) or 'NONE')
print('never cited:', sorted(bibkeys - cited) or 'NONE')
