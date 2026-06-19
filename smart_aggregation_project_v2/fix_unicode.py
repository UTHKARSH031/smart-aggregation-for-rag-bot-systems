"""Fix non-cp1252 unicode characters in all project source files."""
import os
import sys

files = [
    'main code/embeddings.py',
    'main code/smart_aggregation.py',
    'main code/method_b_compression.py',
    'main code/method_c_clustering.py',
    'main code/chunkers.py',
    'main code/evaluation.py',
    'main code/sample_data.py',
    'main code/financebench_loader.py',
    'demo_enhanced.py',
]

replacements = [
    ('\u2713', '[OK]'),
    ('\u2717', '[X]'),
    ('\u2714', '[OK]'),
    ('\u2500', '-'),
    ('\u2550', '='),
    ('\u2551', '|'),
    ('\u2554', '+'),
    ('\u2557', '+'),
    ('\u255a', '+'),
    ('\u255d', '+'),
    ('\u2014', '--'),
    ('\u2013', '-'),
    ('\u2022', '-'),
    ('\u25b6', '>>'),
    ('\u274c', 'ERROR:'),
    ('\u26a0', 'WARN'),
    ('\u2192', '->'),
    ('\u2018', "'"),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
    ('\ufe0f', ''),  # variation selector
]

for fp in files:
    if not os.path.exists(fp):
        print(f'  SKIP (not found): {fp}')
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read()
    modified = False
    for uc, asc in replacements:
        if uc in txt:
            txt = txt.replace(uc, asc)
            modified = True
    if modified:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f'  fixed: {fp}')
    else:
        print(f'  ok   : {fp}')

# Final check
print('\nFinal cp1252 safety check:')
any_bad = False
for fp in files:
    if not os.path.exists(fp):
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read()
    for i, line in enumerate(txt.splitlines(), 1):
        for ci, ch in enumerate(line):
            try:
                ch.encode('cp1252')
            except UnicodeEncodeError:
                print(f'  BAD: {fp} line {i} col {ci} U+{ord(ch):04X}')
                any_bad = True
                break

if not any_bad:
    print('  All files are cp1252-safe!')
