import re

with open('mt_lnn_arxiv.tex', encoding='utf-8') as f:
    src = f.read()

# Check begin/end balance
envs = re.findall(r'\\(begin|end)\{(\w+)\}', src)
stack = []
errors = []
for cmd, env in envs:
    if cmd == 'begin':
        stack.append(env)
    else:
        if not stack or stack[-1] != env:
            errors.append(f'Unmatched end{{{env}}}, stack top: {stack[-1] if stack else "empty"}')
        else:
            stack.pop()
if stack:
    errors.append(f'Unclosed environments: {stack}')
if errors:
    for e in errors:
        print('ERROR:', e)
else:
    print('All environments balanced.')

# Check column spec vs ampersands in tabular rows
tables = re.findall(r'\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}', src, re.DOTALL)
for spec, body in tables:
    ncols = len([c for c in spec if c in 'lcrp'])
    rows = [r for r in body.split('\\\\') if r.strip() and not r.strip().startswith('%')]
    for row in rows:
        amps = row.count('&')
        if amps > 0 and amps != ncols - 1:
            snippet = row.strip()[:80]
            print(f'WARN col mismatch: spec={spec!r} ({ncols} cols) but {amps+1} cells in: {snippet!r}')
print('Column check done.')

# Check \phig is defined
if r'\newcommand{\phig}' in src:
    print('phig macro defined: OK')
else:
    print('ERROR: phig macro missing')
