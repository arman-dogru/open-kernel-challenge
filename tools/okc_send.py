"""Send a committed runs/ file to the judge and record its sha256 at send time (audit r2 m1).
Usage: python tools/okc_send.py check|submit <problem>/runs/<file>.py
Refuses files that are untracked or differ from HEAD. Appends to sends.tsv:
time_et  kind  job_id  problem  file  sha256  git_commit"""
import datetime, hashlib, json, subprocess, sys

kind, path = sys.argv[1], sys.argv[2]
assert kind in ("check", "submit"), kind
assert "/runs/" in path, "send a runs/ file, never submission.py"
if subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True).returncode:
    sys.exit(f"{path} is not committed")
if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path]).returncode:
    sys.exit(f"{path} differs from HEAD; commit it first")
data = open(path, "rb").read()
sha = hashlib.sha256(data).hexdigest()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = subprocess.run(["okc", kind, path, "--no-wait", "--json"], capture_output=True, text=True)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Record every send attempt, even when okc fails or its output doesn't parse (audit r3 nit).
try:
    job = json.loads(out.stdout)
    job_id, problem = job["job_id"], job["problem"]
except (ValueError, KeyError, TypeError):
    job_id, problem = "UNPARSED(rc=%d)" % out.returncode, path.split("/")[0]
same = hashlib.sha256(open(path, "rb").read()).hexdigest() == sha
with open("sends.tsv", "a") as f:
    f.write("\t".join([now, kind, job_id, problem, path, sha if same else sha + "(FILE CHANGED DURING SEND)", commit]) + "\n")
if out.returncode or job_id.startswith("UNPARSED") or not same:
    sys.exit(out.stdout + out.stderr + ("" if same else "\nfile changed during send"))
print(job_id, sha[:16])
