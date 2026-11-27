"""Featurisation of the LigTerm scorer, identical in convention to the development library that fitted the models (protocol v1) for upper-case ASCII sequences, the form of every UniProt sequence of the development runs:
residues are 1-based; bond i lies between residue i and i+1 (P1 = residue i, P1' = residue i+1); the N-terminal bond of a peptide (start, end) is start-1 (needs start >= 2) and its C-terminal bond is end (needs end <= L-1);
a window of w residues on each side is tokenised as P_w..P1, P1'..P_w'; the 20 standard residues are 0..19 and everything else (non-standard letters, positions outside the protein) is the pad token 20.
One DIFFERENCE from the development library, deliberate and tested: lower-case ASCII letters of the 20 standard residues are read as the upper-case residue (a soft-masked FASTA stays usable), where the development library would treat them as unknown; every other character (digits, tabs, non-ASCII characters) is the pad token, one token per character.
Only numpy is needed."""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD = 20                      # pad token: outside the protein, or any character that is not one of the 20 standard residues
PADW = 16                     # padding on each side of a protein = the largest window supported (w <= 16, the largest window of the development library)
NTOK = 21                     # token categories (20 residues + pad)
LUT = np.full(256, PAD, np.int8)
for _i, _a in enumerate(AA):
    LUT[ord(_a)] = _i
    LUT[ord(_a.lower())] = _i


def encode(seq):
    """sequence string -> int8 tokens, one per character: the 20 standard residues (upper or lower case ASCII) are 0..19, every other character is PAD (a non-ASCII character is first replaced by '?', so the length never changes)."""
    return LUT[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]


def padded(seq):
    r = encode(seq)
    pad = np.full(PADW, PAD, np.int8)
    return np.concatenate([pad, r, pad])


def window_tokens(seq, w):
    """tokens P_w..P1, P1'..P_w' of every bond 1..L-1 of `seq`: int8 array (L-1, 2w); empty for L < 2."""
    if not 1 <= w <= PADW:
        raise ValueError(f"window w must be in 1..{PADW}, got {w}")
    L = len(seq)
    if L < 2:
        return np.empty((0, 2 * w), np.int8)
    R = padded(seq)
    i = np.arange(1, L)
    ctr = PADW + i - 1                                   # index of residue i (P1) in the padded array
    return R[ctr[:, None] + np.arange(-w + 1, w + 1)[None, :]]


def linear_score(tok, W, b):
    """logit of a one-hot logistic window model: b + sum over the 2w positions of W[position, token]; tok (n, 2w), W (2w, 21)."""
    t = tok.astype(np.int64)
    return b + W[np.arange(W.shape[0])[None, :], t].sum(1)


def read_fasta(path_or_lines):
    """minimal FASTA reader: yields (id, sequence); id = first whitespace-delimited token of the header; whitespace at the ends of a line, spaces inside a line and '*' are dropped, any other character (digits, a tab inside a line) is kept and becomes the pad token; empty sequences are skipped; duplicate ids are an error."""
    if isinstance(path_or_lines, (str, bytes)) and not isinstance(path_or_lines, list):
        with open(path_or_lines, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = list(path_or_lines)
    seen, name, buf = set(), None, []

    def flush():
        if name is not None and buf:
            yield name, "".join(buf)

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            yield from flush()
            name = ln[1:].split()[0] if ln[1:].split() else ""
            if name in seen:
                raise ValueError(f"duplicate FASTA identifier {name!r}")
            seen.add(name)
            buf = []
        else:
            if name is None:
                raise ValueError("sequence before the first FASTA header")
            buf.append(ln.replace(" ", "").replace("*", ""))
    yield from flush()
