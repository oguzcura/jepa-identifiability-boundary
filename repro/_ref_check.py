import re

tex = open('paper/main.tex', encoding='utf-8').read()

# collect labels defined
defined = set(re.findall(r'\\label\{([^}]*)\}', tex))
# collect references used
used = set()
for m in re.finditer(r'\\(?:ref|eqref|autoref|cref)\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        used.add(k.strip())
for m in re.finditer(r'\\cite[tp]?\*?\{([^}]*)\}', tex):
    pass

print('labels defined:', sorted(defined))
print('refs used     :', sorted(used))
print('DANGLING refs :', sorted(used - defined) or 'NONE')
print('unused labels :', sorted(defined - used) or 'NONE')
print('tables:', len(re.findall(r'\\begin\{table', tex)),
      '| figures:', len(re.findall(r'\\begin\{figure', tex)))
