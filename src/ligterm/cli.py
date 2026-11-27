"""Command line of LigTerm: `python -m ligterm score --fasta in.fa --model MODELDIR --out OUTDIR [--top-peptides K] [--batch-bonds N]` (or the same with `python -m ligterm.cli`).
The model family (the ensemble LigTerm 'ens', the boosted LigTerm-GB 'gb', or a linear reference model) is read from the meta.json of the model directory. Writes bonds.tsv (every bond of every protein with its N and C score and rank),
peptides.tsv (the K best candidate peptides of each protein) and run.json (the model family, the tree counts of a boosted model, the number of seeds of the window MLP of the ensemble, and the wall and CPU seconds). CPU only; numpy only.
Proteins are scored in batches of at most --batch-bonds bonds (default 100,000) because the tree evaluation of the boosted model costs a fixed amount per call; the scores do not depend on the batching."""
import argparse
import json
import os
import sys
import time
import numpy as np

from .features import read_fasta
from .model import competition_rank, peptide_candidates
from .gb import load_model

NOTE = ("scores are logits that rank bonds and peptides within a protein; they are not probabilities and not comparable across proteins; 'N_score' of bond i = peptide starting at residue i+1, 'C_score' of bond i = peptide ending at residue i; "
        "bond ranks are competition ranks (rank 1 = highest score, tied scores share the best rank); peptide ranks are ordinal positions in the written order (ties keep the order of increasing length, then of increasing start); "
        "a measure of observed ligand-terminal propensity, not of proteolytic cleavage")


def run(fasta, model_dir, out_dir, top_peptides=20, batch_bonds=100_000):
    t_wall, t_cpu = time.time(), time.process_time()
    m = load_model(model_dir)
    family = getattr(m, "family", "linear")
    os.makedirs(out_dir, exist_ok=True)
    counts = dict(nb=0, npep=0, nprot=0)
    skipped = []

    def score_batch(batch, fb, fp):
        seqs = [s for _, s in batch]
        scores = m.bond_scores_many(seqs) if hasattr(m, "bond_scores_many") else [m.bond_scores(s) for s in seqs]
        for (pid, seq), (sN, sC) in zip(batch, scores):
            L = len(seq)
            rN, rC = competition_rank(sN), competition_rank(sC)
            for k in range(L - 1):
                fb.write(f"{pid}\t{k + 1}\t{seq[k]}\t{seq[k + 1]}\t{sN[k]:.6f}\t{rN[k]}\t{sC[k]:.6f}\t{rC[k]}\n")
            counts["nb"] += L - 1
            counts["nprot"] += 1
            if top_peptides:
                st, en, ln, sc = peptide_candidates(sN, sC, L, m.len_lo, m.len_hi, m.len_logp, top=top_peptides)
                for j in range(len(st)):
                    fp.write(f"{pid}\t{st[j]}\t{en[j]}\t{ln[j]}\t{sc[j]:.6f}\t{j + 1}\n")
                counts["npep"] += len(st)

    with open(os.path.join(out_dir, "bonds.tsv"), "w", encoding="utf-8") as fb, open(os.path.join(out_dir, "peptides.tsv"), "w", encoding="utf-8") as fp:
        fb.write("protein\tbond\tP1\tP1prime\tN_score\tN_rank\tC_score\tC_rank\n")
        fp.write("protein\tstart\tend\tlength\tscore\trank\n")
        batch, nbatch = [], 0
        for pid, seq in read_fasta(fasta):
            if len(seq) < 2:
                skipped.append(pid)
                continue
            batch.append((pid, seq))
            nbatch += len(seq) - 1
            if nbatch >= batch_bonds:
                score_batch(batch, fb, fp)
                batch, nbatch = [], 0
        if batch:
            score_batch(batch, fb, fp)
    info = dict(proteins=counts["nprot"], bonds=counts["nb"], peptides_written=counts["npep"], window=m.w, model_family=family, model_dir=os.path.abspath(model_dir), numpy=np.__version__,
                skipped_shorter_than_2_residues=skipped, wall_seconds=round(time.time() - t_wall, 3), cpu_seconds=round(time.process_time() - t_cpu, 3), note=NOTE)
    if family == "gb":
        info["n_trees"] = m.n_trees
    if family == "ens":
        info["n_trees"] = m.gb.n_trees
        info["mlp_seeds"] = m.n_seeds
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ligterm", description="score every N and C terminal bond of the proteins of a FASTA file with LigTerm (ensemble), LigTerm-GB (boosted) or a linear reference model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--fasta", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--top-peptides", type=int, default=20, help="candidate peptides written per protein (0 = none)")
    sc.add_argument("--batch-bonds", type=int, default=100_000, help="proteins are scored in batches of at most this many bonds (memory and speed only; the scores do not depend on it)")
    a = ap.parse_args(argv)
    info = run(a.fasta, a.model, a.out, a.top_peptides, a.batch_bonds)
    print(json.dumps({k: info[k] for k in ("proteins", "bonds", "peptides_written", "model_family", "wall_seconds", "cpu_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
