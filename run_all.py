"""Runs the whole pipeline in order: analysis -> report -> slides.

Use this rather than calling the build scripts directly. They read whatever
results.json happens to be on disk, so building without re-running the analysis
first is how a stale number ends up in the report.
"""
import subprocess
import sys

STEPS = [
    ("analysis.py", "fits the models, writes results.json and figures/"),
    ("build_report.py", "writes report.html"),
    ("build_slides.py", "writes presentation.html"),
]

for script, what in STEPS:
    print(f"\n=== {script} -- {what} ===", flush=True)
    r = subprocess.run([sys.executable, script])
    if r.returncode:
        sys.exit(f"{script} failed (exit {r.returncode}); stopping.")

print("\nDone. Deliverables: report.html, presentation.html")
