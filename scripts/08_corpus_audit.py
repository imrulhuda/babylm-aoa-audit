"""Compare training-corpus word frequencies against real infant input
frequencies from the SEEDLingS noun annotations."""

"""Ecological-validity check against the SEEDLingS annotated-nouns corpus.

Compares the paper's CDI-derived measures against real infant input statistics
from the Bergelson lab's public seedlings-nouns dataset (44 infants, months
6-17, 358,300 annotated noun tokens; Bergelson et al. 2019, Dev. Sci.;
Bergelson & Aslin 2017, PNAS; CC-BY-4.0,
https://github.com/BergelsonLab/seedlings-nouns).

Reported in the paper (Discussion + Limitations):
  1. Spearman correlation between our training-corpus word frequency and real
     SEEDLingS input frequency, over CDI words matched by lemma.
  2. Spearman correlation between human CDI AoA and real SEEDLingS input
     frequency vs. the same against training-corpus frequency.

Inclusion: CDI words from the main analysis (aoa_trajectory.json) whose lemma
matches a seedlings-nouns `global_basic_level` lemma with >= 10 tokens.

Usage:
  python 07_seedlings_ecological_check.py \
      --trajectory idea1_aoa_curriculum/results/aoa_trajectory.json \
      --seedlings_csv idea1_aoa_curriculum/data/seedlings-nouns/seedlings-nouns.csv \
      --out idea1_aoa_curriculum/results/seedlings_ecological_check.json
"""
import argparse
import csv
import json
import math
from collections import Counter

import numpy as np
from scipy import stats

MIN_TOKENS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="idea1_aoa_curriculum/results/aoa_trajectory.json")
    ap.add_argument("--seedlings_csv",
                    default="idea1_aoa_curriculum/data/seedlings-nouns/seedlings-nouns.csv")
    ap.add_argument("--out", default="idea1_aoa_curriculum/results/seedlings_ecological_check.json")
    args = ap.parse_args()

    traj = json.load(open(args.trajectory))

    tok_count = Counter()
    with open(args.seedlings_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tok_count[row["global_basic_level"].strip().lower()] += 1

    rows = []
    for w, d in traj["per_word"].items():
        if tok_count.get(w, 0) >= MIN_TOKENS:
            rows.append((w, d["human_aoa_months"], math.log(d["corpus_freq"]),
                         math.log(tok_count[w])))

    words = [r[0] for r in rows]
    human = np.array([r[1] for r in rows], dtype=float)
    corp = np.array([r[2] for r in rows])
    sdl = np.array([r[3] for r in rows])

    def sp(a, b):
        r, p = stats.spearmanr(a, b)
        return {"rho": float(r), "p": float(p)}

    out = {
        "description": "Ecological check of CDI/corpus measures against SEEDLingS "
                       "real infant input frequency (nouns matched by lemma, "
                       f">={MIN_TOKENS} SEEDLingS tokens)",
        "n_words": len(rows),
        "n_paper_words_total": len(traj["per_word"]),
        "corpus_logfreq_vs_seedlings_logfreq": sp(corp, sdl),
        "human_aoa_vs_seedlings_logfreq": sp(human, sdl),
        "human_aoa_vs_corpus_logfreq_same_subset": sp(human, corp),
        "words": words,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"n = {out['n_words']} matched CDI nouns "
          f"(of {out['n_paper_words_total']} analyzed words)")
    for k in ("corpus_logfreq_vs_seedlings_logfreq",
              "human_aoa_vs_seedlings_logfreq",
              "human_aoa_vs_corpus_logfreq_same_subset"):
        print(f"{k}: rho={out[k]['rho']:.3f} p={out[k]['p']:.3g}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
