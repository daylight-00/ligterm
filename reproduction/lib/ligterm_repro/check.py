"""Comparison of regenerated results with the expected tables (`reproduction/expected/`).

Cell-based experiments write one record per study (`study_<pmid>.json`) with a list of cells; `cell_rows` flattens the cells to the rows of an expected table (key columns, counts and the
metrics with their intervals), `read_table` and `write_table` move a table to and from a gzip TSV with exact float text (repr), and `compare_cells` compares two tables under the policy of the
package: every value equal, with equal types, except the retrained NetCleave-style networks (a torch run is not bitwise reproducible across machines), whose values may differ by at most 1e-4.
`compare_json` compares two JSON documents value by value with an optional absolute level for floats (a NaN equals a NaN). Nothing here depends on the experiment code.
"""
import glob
import gzip
import json
import math
import os

KEY_COLUMNS = ("study", "terminus", "analysis", "kind", "model", "fit", "model_a", "fit_a", "model_b", "fit_b")
COUNT_COLUMNS = ("n_proteins", "n_groups", "n_pos")
METRIC_COLUMNS = ("bond_auroc", "bond_auroc_lo", "bond_auroc_hi", "bond_auprc", "bond_auprc_lo", "bond_auprc_hi", "bond_recall_at_10pct", "bond_recall_at_10pct_lo", "bond_recall_at_10pct_hi",
                  "local_auc", "local_auc_lo", "local_auc_hi", "local_auc_interior", "local_auc_interior_lo", "local_auc_interior_hi", "local_auc_outward", "local_auc_outward_lo", "local_auc_outward_hi")
COLUMNS = KEY_COLUMNS + COUNT_COLUMNS + METRIC_COLUMNS
NETWORK_PREFIX = "NetCleave-style"
NETWORK_LEVEL = 1e-4


def cell_rows(results_dir):
    """the rows of the expected table of one cell-based experiment: every cell of every study file of the folder, in study and cell order"""
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "study_*.json"))):
        for c in json.load(open(f))["cells"]:
            rows.append({k: c.get(k) for k in COLUMNS})
    return rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, str)):
        return str(v)
    return repr(v)                                                   # int and float: exact text


def write_table(rows, path):
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:      # no file name and no time stamp in the header: the same table gives the same bytes
        g.write(("\t".join(COLUMNS) + "\n").encode())
        for r in rows:
            g.write(("\t".join(_fmt(r[k]) for k in COLUMNS) + "\n").encode())


def read_table(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        head = next(fh).rstrip("\n").split("\t")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            r = {}
            for k, v in zip(head, f):
                if v == "":
                    r[k] = None
                elif k in COUNT_COLUMNS:
                    r[k] = int(v)
                elif k in METRIC_COLUMNS:
                    r[k] = float(v)
                else:
                    r[k] = v
            rows.append(r)
    return rows


def row_key(r):
    return tuple(r[k] for k in KEY_COLUMNS)


def is_network(r):
    return any(str(r.get(k) or "").startswith(NETWORK_PREFIX) for k in ("model", "model_a", "model_b"))


def nonfinite(rows):
    return sum(1 for r in rows for k in METRIC_COLUMNS if isinstance(r[k], float) and not math.isfinite(r[k]))


def compare_cells(new, exp, subset=False):
    """-> dict of counts and a list of problems. subset: the new rows may cover only some of the studies of the expected table (a partial re-run)."""
    problems = []
    studies_exp = {r["study"] for r in exp}
    studies_new = {r["study"] for r in new}
    if not subset and studies_new != studies_exp:
        problems.append(f"studies: missing {sorted(studies_exp - studies_new)}, extra {sorted(studies_new - studies_exp)}")
    if studies_new - studies_exp:
        problems.append(f"studies not in the expected table: {sorted(studies_new - studies_exp)}")
    nk = [row_key(r) for r in new]
    if len(set(nk)) != len(nk):
        problems.append(f"{len(nk) - len(set(nk))} repeated cell keys in the regenerated results")
    em = {row_key(r): r for r in exp}
    nm = {row_key(r): r for r in new}
    in_scope = {k for k in em if k[0] in studies_new} if subset else set(em)
    miss, extra = sorted(in_scope - set(nm), key=str), sorted(set(nm) - set(em), key=str)
    if miss:
        problems.append(f"{len(miss)} expected cells are missing, e.g. {miss[0]}")
    if extra:
        problems.append(f"{len(extra)} cells are not expected, e.g. {extra[0]}")
    n_cmp = n_equal = n_net = 0
    net_max = 0.0
    for k in set(nm) & set(em):
        a, b = nm[k], em[k]
        n_cmp += 1
        d = 0.0
        ok = True
        for c in COUNT_COLUMNS + METRIC_COLUMNS:
            x, y = a[c], b[c]
            if x is None and y is None:
                continue
            if type(x) is not type(y):
                ok = False
                problems.append(f"{k} {c}: {x!r} against {y!r}")
                continue
            if isinstance(x, float):
                if math.isnan(x) and math.isnan(y):
                    continue
                if x != y:
                    d = max(d, abs(x - y)) if math.isfinite(x - y) else float("inf")
                    if not (is_network(a) and abs(x - y) <= NETWORK_LEVEL):
                        ok = False
                        problems.append(f"{k} {c}: {x!r} against {y!r} (difference {abs(x - y):.3e})")
            elif x != y:
                ok = False
                problems.append(f"{k} {c}: {x!r} against {y!r}")
        n_equal += ok
        if is_network(a):
            n_net += 1
            net_max = max(net_max, d)
    nf_new, nf_exp = nonfinite(new), nonfinite([r for r in exp if r["study"] in studies_new])
    if nf_new != nf_exp:
        problems.append(f"non-finite values: {nf_new} in the regenerated results, {nf_exp} expected")
    return dict(cells=len(new), compared=n_cmp, equal=n_equal, network_cells=n_net, network_max_difference=net_max, problems=problems)


def _diffs(a, b, path, level):
    if type(a) is not type(b):
        return [f"{path}: {a!r} against {b!r}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in only one of the two")
            else:
                out += _diffs(a[k], b[k], f"{path}.{k}", level)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} against {len(b)} entries"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diffs(x, y, f"{path}[{i}]", level)
        return out
    if isinstance(a, float):
        if (math.isnan(a) and math.isnan(b)) or a == b:
            return []
        return [] if abs(a - b) <= level else [f"{path}: {a!r} against {b!r} (difference {abs(a - b):.3e})"]
    return [] if a == b else [f"{path}: {a!r} against {b!r}"]


def compare_json(new_path, exp_path, level=0.0):
    """-> list of differences between two JSON documents (an empty list: equal). level: the absolute difference allowed in a float; every other value must be equal."""
    return _diffs(json.load(open(new_path)), json.load(open(exp_path)), "$", level)
