"""Repeat the corpus-versus-infant-input comparison separately for each source
corpus."""

import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats

SOURCES = ["childes", "bnc_spoken", "switchboard", "gutenberg",
           "open_subtitles", "simple_wiki"]
FILES = {s: f"{s}.train.txt" for s in SOURCES}
SPEAKER = re.compile(r"^\*[A-Za-z0-9]{1,8}:\s*")
TOKRE = re.compile(r"[a-z']+")
MIN_TOKENS = 10          # same threshold as the published audit


def sentence_counts(path, words):
    """#sentences in this file containing each word (speaker codes stripped)."""
    c = Counter()
    with open(path) as f:
        for line in f:
            txt = SPEAKER.sub("", line.strip()).lower()
            if not txt:
                continue
            for w in set(TOKRE.findall(txt)) & words:
                c[w] += 1
    return c


def sp(a, b):
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--aoa_norms", required=True)
    ap.add_argument("--seedlings_csv", required=True)
    ap.add_argument("--trajectory", required=True,
                    help="supplies the analysed word set + human AoA")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    traj = json.load(open(a.trajectory))["per_word"]
    words = set(traj)
    human = {w: traj[w]["human_aoa_months"] for w in words}

    # SEEDLingS real-infant-input frequency (lemma token counts)
    sdl = Counter()
    with open(a.seedlings_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sdl[row["global_basic_level"].strip().lower()] += 1

    out = {"label": a.label, "data_dir": a.data_dir, "min_tokens": MIN_TOKENS,
           "n_words_in_trajectory": len(words), "sources": {}}

    per_source = {}
    for s in SOURCES:
        p = Path(a.data_dir) / FILES[s]
        if not p.exists():
            print(f"  {s}: MISSING"); continue
        per_source[s] = sentence_counts(p, words)

    # ALL = pooled across sources, for a like-for-like reference row
    pooled = Counter()
    for c in per_source.values():
        pooled.update(c)

    print(f"{'source':<16}{'n':>5}  {'corpus~SEEDL':>13}  {'AoA~SEEDL':>11}  {'AoA~corpus':>11}")
    print("-" * 62)
    for name, cnt in list(per_source.items()) + [("ALL", pooled)]:
        keys = [w for w in words
                if cnt.get(w, 0) > 0 and sdl.get(w, 0) >= MIN_TOKENS]
        if len(keys) < 20:
            out["sources"][name] = {"n": len(keys), "note": "too few matched words"}
            print(f"{name:<16}{len(keys):>5}  (too few matched words)")
            continue
        cf = np.log([cnt[w] for w in keys])
        sf = np.log([sdl[w] for w in keys])
        hu = np.array([human[w] for w in keys], float)
        r1, p1 = sp(cf, sf); r2, p2 = sp(hu, sf); r3, p3 = sp(hu, cf)
        out["sources"][name] = {
            "n": len(keys),
            "corpus_vs_seedlings": {"rho": r1, "p": p1},
            "human_aoa_vs_seedlings": {"rho": r2, "p": p2},
            "human_aoa_vs_corpus": {"rho": r3, "p": p3},
        }
        print(f"{name:<16}{len(keys):>5}  {r1:>13.3f}  {r2:>11.3f}  {r3:>11.3f}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
