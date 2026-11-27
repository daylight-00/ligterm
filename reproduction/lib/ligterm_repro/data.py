"""Data and window conventions of the terminal-bond analyses.

Residues are 1-based; bond i lies between residue i and i+1 (P1 = residue i, P1' = residue i+1).
The N-bond of a unit (start, end) is start-1 (needs start >= 2), its C-bond is end (needs end <= L-1).
Token matrix of a set of bonds: columns 0..15 are P16..P1, columns 16..31 are P1'..P16'; token 20 is the pad
(outside the protein, or a non-standard residue).
"""
import collections
import csv
import gzip
import json

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, PADW, KEY, WMAX = 20, 16, 70000, 16
LUT = np.full(256, PAD, np.int8)
for _i, _a in enumerate(AA):
    LUT[ord(_a)] = _i


class Ctx:
    pass


def load_ctx(units_tsv, seqs_json, groups_tsv=None):
    """Verified units and the residue array.

    units_tsv: the units table (acc, start, end, seq, pmids; gzip); seqs_json: accession to UniProt sequence;
    groups_tsv: the sequence groups of every protein (columns g_<code>). A unit is verified if its peptide is the
    stretch start..end of the UniProt sequence."""
    seqs = json.load(open(seqs_json))
    U = []
    csv.field_size_limit(10**8)
    with gzip.open(units_tsv, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            U.append((r["acc"], int(r["start"]), int(r["end"]), r["seq"], tuple(r["pmids"].split(","))))
    ver = np.array([bool(seqs.get(u[0])) and seqs[u[0]][u[1] - 1:u[2]] == u[3] for u in U])
    c = build_ctx([U[i] for i in np.flatnonzero(ver)], seqs)
    if groups_tsv:
        with gzip.open(groups_tsv, "rt") as fh:
            head = next(fh).rstrip("\n").split("\t")
            rows = [ln.rstrip("\n").split("\t") for ln in fh]
        assert [r[0] for r in rows] == c.names, "group file does not follow the protein order"
        for j, nm in enumerate(head[1:], start=1):
            c.G[nm[2:]] = np.array([int(r[j]) for r in rows], np.int64)
    return c


def build_ctx(V, seqs):
    """V = list of verified units (acc, start, end, peptide, pmids tuple); seqs = accession to sequence."""
    names = sorted({u[0] for u in V})
    c = Ctx()
    c.seqs, c.names, c.idx = seqs, names, {a: i for i, a in enumerate(names)}
    c.NP = len(names)
    c.L = np.array([len(seqs[a]) for a in names], np.int64)
    off = np.zeros(c.NP, np.int64)
    parts = [np.full(PADW, PAD, np.int8)]
    pos = PADW
    for i, a in enumerate(names):
        off[i] = pos
        parts.append(LUT[np.frombuffer(seqs[a].encode(), np.uint8)])
        parts.append(np.full(PADW, PAD, np.int8))
        pos += len(seqs[a]) + PADW
    c.off, c.R = off, np.concatenate(parts)
    c.vp = np.array([c.idx[u[0]] for u in V], np.int64)
    c.vs = np.array([u[1] for u in V], np.int64)
    c.ve = np.array([u[2] for u in V], np.int64)
    c.vseq = [u[3] for u in V]
    c.vpm = [u[4] for u in V]
    c.nv = len(V)
    stud = collections.defaultdict(list)
    for r_, pms in enumerate(c.vpm):
        for p in pms:
            stud[p].append(r_)
    c.stud = {k: np.array(v, np.int64) for k, v in stud.items()}
    c.G = {}
    return c


def bond_of(c, rows, T):
    """terminal bond of the verified units `rows` for terminus T: (valid mask, bond index)."""
    if T not in ("N", "C"):
        raise ValueError(f"terminus must be 'N' or 'C', got {T!r}")
    L = c.L[c.vp[rows]]
    if T == "N":
        return c.vs[rows] >= 2, c.vs[rows] - 1
    return c.ve[rows] <= L - 1, c.ve[rows]


def tokens(c, p, i, w=WMAX):
    """(n, 2w) int8 residue tokens P_w..P1, P1'..P_w' of bonds (protein index p, 1-based bond i); pad outside the protein."""
    ctr = c.off[p] + i - 1
    return c.R[ctr[:, None] + np.arange(-w + 1, w + 1)[None, :]]


def posfeat(i, L):
    """position features of a bond (index i, protein length L), used by the position-only model."""
    i = i.astype(np.float64)
    L = L.astype(np.float64)
    return np.stack([np.log1p(i), np.log1p(L - i), i / L, (i / L) ** 2, (i <= 30).astype(float), ((L - i) <= 30).astype(float)], 1)
