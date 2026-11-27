#!/usr/bin/env python3
"""Check regenerated results against the expected tables of the package.

`expected/CHECKS.json` lists the checks; each names a result in the layout of the `results/` folder of the data package (what the scripts write, one sub-folder per experiment) and its expected table:
  cells  an experiment with one record per study: every cell (study, terminus, analysis, model, fit, counts, metrics with intervals) must be present exactly once and equal, with equal types;
         only the retrained NetCleave-style networks may differ, by at most 1e-4 (a torch run is not bitwise reproducible across machines); the studies must be those of the expected table
         (--subset accepts a partial re-run: only the studies that were re-run are compared); the number of non-finite values must be the same;
  json   a result document compared value by value, with the level of the check (0 = equal; a table written with fewer digits than the computation has a small stated level);
  csv, text  a table or a report compared byte by byte.
A check whose result is not there is reported as NOT RUN (an error with --require-all). Exit code 0 only if nothing failed.

usage: 90_check_reproduction.py --results <results folder> [--expected <expected folder>] [--only NAME ...] [--subset] [--require-all]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import check  # noqa: E402


def run_check(spec, results, expected, subset):
    kind = spec["kind"]
    res_path = os.path.join(results, spec["results"])
    exp_path = os.path.join(expected, spec["expected"])
    if not os.path.exists(res_path):
        return "NOT RUN", [f"{spec['results']} does not exist"]
    if kind == "cells":
        new, exp = check.cell_rows(res_path), check.read_table(exp_path)
        r = check.compare_cells(new, exp, subset=subset)
        line = f"{r['cells']} cells, {r['equal']} of {r['compared']} compared equal; network cells {r['network_cells']} (largest difference {r['network_max_difference']:.1e})"
        return ("PASS" if not r["problems"] else "FAIL"), [line] + r["problems"][:10] + ([f"... and {len(r['problems']) - 10} more"] if len(r["problems"]) > 10 else [])
    if kind == "json":
        d = check.compare_json(res_path, exp_path, spec.get("level", 0.0))
        return ("PASS" if not d else "FAIL"), ([f"equal (level {spec.get('level', 0.0):g})"] if not d else d[:10] + ([f"... and {len(d) - 10} more"] if len(d) > 10 else []))
    if kind in ("csv", "text"):
        same = open(res_path, "rb").read() == open(exp_path, "rb").read()
        return ("PASS" if same else "FAIL"), ["byte-equal" if same else "the bytes differ"]
    raise ValueError(f"unknown kind {kind!r}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--expected", default=os.path.join(here, "..", "expected"))
    ap.add_argument("--only", nargs="+", help="names of the checks to run (default: all)")
    ap.add_argument("--subset", action="store_true", help="the cell-based results may cover only some of the studies")
    ap.add_argument("--require-all", action="store_true", help="a check that was not run is an error")
    a = ap.parse_args()
    checks = json.load(open(os.path.join(a.expected, "CHECKS.json")))["checks"]
    if a.only:
        unknown = set(a.only) - {c["name"] for c in checks}
        if unknown:
            sys.exit(f"unknown checks: {sorted(unknown)}")
        checks = [c for c in checks if c["name"] in set(a.only)]
    counts = {"PASS": 0, "FAIL": 0, "NOT RUN": 0}
    for spec in checks:
        status, lines = run_check(spec, a.results, a.expected, a.subset)
        counts[status] += 1
        print(f"{status:7s} {spec['name']}: {lines[0]}")
        for ln in lines[1:]:
            print(f"        {ln}")
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['NOT RUN']} not run")
    return 1 if counts["FAIL"] or (a.require_all and counts["NOT RUN"]) else 0


if __name__ == "__main__":
    sys.exit(main())
