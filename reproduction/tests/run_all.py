#!/usr/bin/env python3
"""Runs every test file of this folder in its own process and reports. usage: python3 tests/run_all.py   (numpy and scipy only; about a minute)
The window-audit test checks the window geometry against the scorer's own code when the `ligterm` package of the repository is next to `reproduction/`."""
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
failed = []
for f in sorted(glob.glob(os.path.join(here, "test_*.py"))):
    r = subprocess.run([sys.executable, "-B", f], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(("PASS " if r.returncode == 0 else "FAIL ") + os.path.basename(f) + ": " + last)
    if r.returncode:
        failed.append(os.path.basename(f))
        print(r.stdout[-1500:] + r.stderr[-1500:])
print(f"{'all tests passed' if not failed else 'failed: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
