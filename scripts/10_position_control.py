"""Measure how much of the curriculum's AoA alignment is attributable to the
order in which words are presented during training."""

import argparse, json, re
import numpy as np
from scipy import stats

TOKRE = re.compile(r"[a-z']+")

def partial(a, b, ctrl):
    ra, rb, rc = (stats.rankdata(x) for x in (a, b, ctrl))
    res = lambda y, x: y - np.poly1d(np.polyfit(x, y, 1))(x)
    r, p = stats.pearsonr(res(ra, rc), res(rb, rc))
    return float(r), float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--traj", nargs="+", required=True, help="curriculum seed jsons")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = [json.load(open(f))["per_word"] for f in a.traj]
    words = set(runs[0])
    for r in runs[1:]:
        words &= set(r)

    pos = {w: [] for w in words}
    n = 0
    for line in open(a.corpus):
        ws = TOKRE.findall(line.lower())
        for w in set(ws) & words:
            pos[w].append(n)
        n += len(ws)
    medpos = {w: (np.median(v) / n if v else np.nan) for w, v in pos.items()}

    keys = [w for w in sorted(words)
            if not np.isnan(medpos[w])
            and all(r[w].get("model_aoa_M_words") is not None for r in runs)]
    mp    = np.array([medpos[w] for w in keys])
    human = np.array([runs[0][w]["human_aoa_months"] for w in keys], float)
    model = np.array([[r[w]["model_aoa_M_words"] for w in keys] for r in runs], float).mean(axis=0)

    r_enc, p_enc = stats.spearmanr(mp, human)
    r_fol, p_fol = stats.spearmanr(model, mp)
    r_tot, p_tot = stats.spearmanr(model, human)
    r_res, p_res = partial(model, human, mp)

    out = {"label": a.label, "n_words": len(keys),
           "curriculum_encodes_human_aoa": {"rho": float(r_enc), "p": float(p_enc)},
           "model_follows_feeding_order":  {"rho": float(r_fol), "p": float(p_fol)},
           "model_vs_human_total":         {"rho": float(r_tot), "p": float(p_tot)},
           "residual_controlling_position":{"rho": r_res, "p": p_res}}
    print(f"=== {a.label}  (n={len(keys)}, seeds averaged) ===")
    print(f"  curriculum encodes human AoA   : rho = {r_enc:+.3f}")
    print(f"  model follows feeding order    : rho = {r_fol:+.3f}")
    print(f"  model vs human AoA (total)     : rho = {r_tot:+.3f}")
    print(f"  RESIDUAL controlling position  : rho = {r_res:+.3f}  p = {p_res:.3g}")
    json.dump(out, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
