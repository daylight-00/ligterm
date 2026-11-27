"""Pieces of the local-window similarity audit of the external test cohort that can be tested without the data. numpy only.
The scorer reads a window of 32 residues around a bond: P16..P1 and P1'..P16' = residues b-15..b+16 (1-based) of bond b, which is s[b-16:b+16] for the 0-based sequence s. The audit compares such windows with the windows of the training proteins by the UNGAPPED identity: the number of identical residues at the 32 aligned positions (or at the central 16, P8..P8'). A non-standard residue never matches.
The scan is exact (every training window is compared with every query); no seed, no k-mer filter."""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
W = 32
CENTRAL = (8, 24)                                   # the positions 8..23 of the 32-window: P8..P1 and P1'..P8'
SEP = 100                                           # separator code between training proteins (never equal to a query code)
TRAIN_OTHER, QUERY_OTHER = 99, 98                   # a non-standard residue is coded differently on the two sides, so it never matches


def encode(seq, other):
    """int8 codes: 0 to 19 for the 20 standard residues, `other` for every other character"""
    lut = np.full(256, other, np.int8)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    return lut[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]


def window_start(b, L):
    """0-based start of the 32-residue window of bond b (1-based, 1 <= b <= L - 1) in a protein of L residues, or None when the window is not fully inside the protein"""
    s = b - 16
    return s if (s >= 0 and s + W <= L) else None


def build_training(seqs, accs):
    """-> (concat, offsets, valid): the training proteins one after the other, each followed by W separators; offsets[acc] = the index of its first residue; valid[s] is True when the window that starts at s lies inside one protein"""
    parts, offsets, pos = [], {}, 0
    for a in accs:
        code = encode(seqs[a], TRAIN_OTHER)
        offsets[a] = pos
        parts += [code, np.full(W, SEP, np.int8)]
        pos += len(code) + W
    concat = np.concatenate(parts)
    c = np.concatenate([[0], np.cumsum(concat == SEP)])
    valid = (c[W:] - c[:-W]) == 0                    # length len(concat) - W + 1
    return concat, offsets, valid


def scan(Q, concat, valid, lo, hi, masks=(), chunk=20000):
    """Q: (q, 32) int8 query windows. Over the windows that start in [lo, hi) and are valid: per query the maximum number of identical positions over all 32 positions (m32) and over the central 16 (m16), and the start of the window that gives m32 (the smallest start on a tie within a chunk, the earlier chunk on a tie across chunks).
    For each boolean mask (indexed by window start) the same two maxima over the masked windows (mk32[i], mk16[i]; 0 when the mask selects no window)."""
    q = len(Q)
    out = {"m32": np.zeros(q, np.int16), "a32": np.full(q, -1, np.int64), "m16": np.zeros(q, np.int16),
           "mk32": [np.zeros(q, np.int16) for _ in masks], "mk16": [np.zeros(q, np.int16) for _ in masks]}
    ar = np.arange(q)
    for s0 in range(lo, hi, chunk):
        s1 = min(hi, s0 + chunk)
        c32 = np.zeros((q, s1 - s0), np.int16)
        c16 = np.zeros((q, s1 - s0), np.int16)
        for j in range(W):
            m = concat[s0 + j: s1 + j][None, :] == Q[:, j][:, None]
            c32 += m
            if CENTRAL[0] <= j < CENTRAL[1]:
                c16 += m
        v = valid[s0:s1]
        if not v.all():
            c32[:, ~v] = 0
            c16[:, ~v] = 0
        k = c32.argmax(axis=1)
        mx = c32[ar, k]
        better = mx > out["m32"]
        out["m32"][better] = mx[better]
        out["a32"][better] = s0 + k[better]
        out["m16"] = np.maximum(out["m16"], c16.max(axis=1))
        for i, mk in enumerate(masks):
            sel = mk[s0:s1] & v
            if sel.any():
                out["mk32"][i] = np.maximum(out["mk32"][i], c32[:, sel].max(axis=1))
                out["mk16"][i] = np.maximum(out["mk16"][i], c16[:, sel].max(axis=1))
    return out


def merge(parts):
    """combine the results of scans over disjoint ranges of window starts (in order of increasing start)"""
    res = {"m32": parts[0]["m32"].copy(), "a32": parts[0]["a32"].copy(), "m16": parts[0]["m16"].copy(),
           "mk32": [x.copy() for x in parts[0]["mk32"]], "mk16": [x.copy() for x in parts[0]["mk16"]]}
    for p in parts[1:]:
        better = p["m32"] > res["m32"]
        res["m32"][better] = p["m32"][better]
        res["a32"][better] = p["a32"][better]
        res["m16"] = np.maximum(res["m16"], p["m16"])
        for i in range(len(res["mk32"])):
            res["mk32"][i] = np.maximum(res["mk32"][i], p["mk32"][i])
            res["mk16"][i] = np.maximum(res["mk16"][i], p["mk16"][i])
    return res


def locate(start, offsets_sorted, accs_sorted):
    """(accession, 0-based start inside the protein) of a global window start; offsets_sorted is increasing"""
    i = int(np.searchsorted(offsets_sorted, start, side="right")) - 1
    return accs_sorted[i], int(start - offsets_sorted[i])


def brute_force(Q, concat, valid, lo, hi):
    """the definition written naively (for the tests): (m32, m16) per query over the valid windows with start in [lo, hi)"""
    m32 = np.zeros(len(Q), np.int64)
    m16 = np.zeros(len(Q), np.int64)
    for s in range(lo, hi):
        if not valid[s]:
            continue
        w = concat[s:s + W]
        for i in range(len(Q)):
            eq = Q[i] == w
            m32[i] = max(m32[i], int(eq.sum()))
            m16[i] = max(m16[i], int(eq[CENTRAL[0]:CENTRAL[1]].sum()))
    return m32, m16
