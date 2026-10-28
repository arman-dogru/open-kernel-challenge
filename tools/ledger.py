"""Regenerate the job ledger in STATUS.md from `okc jobs` + runs/profiles (sha256 of the committed runs/ file)."""
import os, re, hashlib, glob, subprocess
txt = subprocess.run(["okc", "jobs"], capture_output=True, text=True).stdout
rows = []
for l in txt.splitlines():
    m = re.match(r'([0-9a-f]{16})\s+(\S+)\s+(check\s+)?(Oct \d+ \d+:\d+)\s+(.)\s+(\w+)\s*(.*)', l)
    if not m:
        continue
    j, p = m.group(1), m.group(2)
    f = glob.glob(f'{p}/runs/profiles/*.{j}.*')
    run = os.path.basename(f[0]).split('.')[0] + '.py' if f else '(profile not saved yet)'
    path = f'{p}/runs/{run}'
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16] if os.path.exists(path) else '—'
    rows.append((m.group(4), j, p, 'check' if m.group(3) else 'SUBMIT', run, sha, m.group(6), m.group(7).strip()))
rows.sort()
head = ["## Job ledger (every okc job; times ET; sha256 = first 16 hex of the runs/ file as committed)", "",
        "Caveat (audit m2): files before the ledger existed were not hashed at send time; `04269b7477f74417`'s",
        "runs/v4_postnorm_fused.py got a docstring-only edit after that check ran; board job `9cb6d6a409ff4cd6`",
        "was sent after the edit, so it matches the committed bytes. From c89f5f5 on, every file is committed before it is sent.", "",
        "| time | job | problem | kind | file | sha256[:16] | status | time |", "|---|---|---|---|---|---|---|---|"]
s = open('STATUS.md').read()
s = s[:s.index("## Job ledger")] if "## Job ledger" in s else s.rstrip() + "\n\n"
open('STATUS.md', 'w').write(s + "\n".join(head + ["| " + " | ".join(r) + " |" for r in rows]) + "\n")
